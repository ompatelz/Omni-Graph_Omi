"""Tests for Access Control Manager and Audit Logging."""
from unittest.mock import MagicMock

import psycopg2
import pytest
from omnigraph.access_control_audit import AccessControlManager


def test_get_user_roles(db_conn):
    """Verify loading assigned roles and permissions for users."""
    acm = AccessControlManager(db_conn)
    # Priya Agarwal (user_id=1) is an admin
    roles = acm.get_user_roles(1)
    assert len(roles) > 0
    role_names = [r["role_name"] for r in roles]
    assert any("admin" in name.lower() for name in role_names)
    assert "permissions" in roles[0]


def test_validate_permission(db_conn):
    """Verify permission check matches assigned role permissions."""
    acm = AccessControlManager(db_conn)
    # Admin has system-level permissions
    assert acm.validate_permission(1, "manage_users") is True
    # Non-existent permission
    assert acm.validate_permission(1, "non_existent_fake_permission") is False
    # Consumer (user_id=8) should not have manage_users
    assert acm.validate_permission(8, "manage_users") is False


def test_user_access_matrix(db_conn):
    """Verify access matrix contains rows for resource types and sensitivity levels."""
    acm = AccessControlManager(db_conn)
    matrix = acm.get_user_access_matrix(1)
    assert len(matrix) > 0
    sensitivities = {m["sensitivity_level"] for m in matrix}
    assert "public" in sensitivities


def test_sensitivity_levels_and_document_access(db_conn):
    """Test access rights across public, internal, confidential, and restricted sensitivity levels."""
    acm = AccessControlManager(db_conn)

    # 1. Admin (user_id=1) has read access to public documents (doc 1)
    assert acm.check_access(user_id=1, resource_type="document", resource_id=1, action="read") is True

    # 2. Consumer (user_id=8) is DENIED read access to restricted document (doc 5)
    assert acm.check_access(user_id=8, resource_type="document", resource_id=5, action="read") is False

    # 3. Compliance officer (user_id=5) is GRANTED read access to restricted document (doc 5)
    assert acm.check_access(user_id=5, resource_type="document", resource_id=5, action="read") is True

    # 4. Consumer (user_id=10) is DENIED write access to confidential document (doc 2)
    assert acm.check_access(user_id=10, resource_type="document", resource_id=2, action="write") is False


def test_check_policy_at_sensitivity(db_conn):
    """Verify explicit policy evaluation at each sensitivity level."""
    acm = AccessControlManager(db_conn)
    # Admin can read and write all 4 sensitivity tiers
    for sens in ("public", "internal", "confidential", "restricted"):
        assert acm.check_policy_at_sensitivity(1, "document", sens, "read") is True

    # Consumer (user_id=8) cannot read restricted
    assert acm.check_policy_at_sensitivity(8, "document", "restricted", "read") is False
    # But can read public
    assert acm.check_policy_at_sensitivity(8, "document", "public", "read") is True


def test_nonexistent_resource_access(db_conn):
    """Access checks for non-existent documents must return False gracefully."""
    acm = AccessControlManager(db_conn)
    assert acm.check_access(user_id=1, resource_type="document", resource_id=999999, action="read") is False


def test_non_document_resource_sensitivity(db_conn):
    """Non-document resources default to public sensitivity."""
    acm = AccessControlManager(db_conn)
    sens = acm._get_resource_sensitivity("concept", 1)
    assert sens == "public"


def test_filter_accessible_documents(db_conn):
    """Verify batch filtering of document IDs according to user permissions."""
    acm = AccessControlManager(db_conn)
    # Doc 1 is public, Doc 5 is restricted
    doc_ids = [1, 5]

    admin_accessible = acm.filter_accessible_documents(user_id=1, doc_ids=doc_ids, action="read")
    assert 1 in admin_accessible
    assert 5 in admin_accessible

    consumer_accessible = acm.filter_accessible_documents(user_id=8, doc_ids=doc_ids, action="read")
    assert 1 in consumer_accessible
    assert 5 not in consumer_accessible

    empty_result = acm.filter_accessible_documents(user_id=1, doc_ids=[], action="read")
    assert empty_result == []


def test_audit_logging_and_trail(db_conn):
    """Verify writing audit log entries and retrieving audit trails."""
    acm = AccessControlManager(db_conn)
    test_details = "Automated pytest audit trail verification"
    audit_id = acm.log_audit(
        user_id=1,
        action="view",
        resource_type="document",
        resource_id=1,
        details=test_details,
        ip_address="127.0.0.1",
    )
    assert audit_id is not None
    assert audit_id > 0

    trail = acm.get_audit_trail(action="view", limit=10)
    assert len(trail) > 0
    assert any(entry["details"] == test_details for entry in trail)


def test_sensitive_access_report(db_conn):
    """Verify sensitive access report executes and returns list."""
    acm = AccessControlManager(db_conn)
    report = acm.get_sensitive_access_report(days=365)
    assert isinstance(report, list)


def test_query_logging_and_analytics(db_conn):
    """Verify query logging and analytics summary aggregation."""
    acm = AccessControlManager(db_conn)
    log_id = acm.log_query(
        user_id=1,
        query_text="pytest query test",
        query_type="keyword_search",
        results_count=4,
        execution_ms=12,
    )
    assert log_id is not None
    assert log_id > 0

    analytics = acm.get_query_analytics(days=30)
    assert "by_type" in analytics
    assert "top_users" in analytics
    assert analytics["period_days"] == 30


def test_role_assignment_and_revocation(db_conn):
    """Test assigning and revoking a role with audit logging."""
    acm = AccessControlManager(db_conn)
    with db_conn.conn.cursor() as cur:
        # Find a valid user and an unassigned role for that user
        cur.execute(
            """
            SELECT u.user_id, r.role_id
            FROM omnigraph.users u
            CROSS JOIN omnigraph.roles r
            LEFT JOIN omnigraph.user_roles ur ON u.user_id = ur.user_id AND r.role_id = ur.role_id
            WHERE ur.role_id IS NULL AND u.user_id != 1
            LIMIT 1
            """
        )
        row = cur.fetchone()
        assert row is not None, "Need at least one unassigned user-role pair"
        target_user_id, target_role_id = row

    assign_ok = acm.assign_role(user_id=target_user_id, role_id=target_role_id, assigned_by=1)
    assert assign_ok is True

    # Revoke it
    revoke_ok = acm.revoke_role(user_id=target_user_id, role_id=target_role_id, revoked_by=1)
    assert revoke_ok is True


def test_access_control_db_error_handling(mock_db):
    """Verify graceful handling when database operations raise psycopg2.Error."""
    acm = AccessControlManager(mock_db)
    cursor = mock_db.conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = psycopg2.OperationalError("DB connection dropped")

    assert acm.check_access(1, "document", 1, "read") is False
    assert acm.validate_permission(1, "read") is False
    assert acm.get_user_roles(1) == []
    assert acm.get_user_access_matrix(1) == []
    assert acm.log_audit(1, "test", "doc", 1) is None
    assert acm.log_query(1, "q", "type") is None
    assert acm.get_audit_trail() == []
    assert acm.get_sensitive_access_report() == []
    assert acm.get_query_analytics() == {}
    assert acm.assign_role(1, 2, 1) is False
    assert acm.revoke_role(1, 2, 1) is False
