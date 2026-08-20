"""The public bucket policy. A write grant here would let anyone alter the corpus."""

from finvec.s3_setup import PUBLIC_ACCESS_BLOCK, public_read_policy


def _actions(policy):
    out = []
    for stmt in policy["Statement"]:
        action = stmt["Action"]
        out += action if isinstance(action, list) else [action]
    return out


def test_policy_grants_read_only():
    actions = _actions(public_read_policy("b"))
    assert "s3:GetObject" in actions
    assert "s3:ListBucket" in actions
    assert not any(a.startswith(("s3:Put", "s3:Delete")) for a in actions), actions
    assert "s3:*" not in actions


def test_policy_is_anonymous_and_scoped_to_the_one_bucket():
    policy = public_read_policy("finvec-corpus")
    assert all(stmt["Principal"] == "*" for stmt in policy["Statement"])
    resources = [stmt["Resource"] for stmt in policy["Statement"]]
    assert "arn:aws:s3:::finvec-corpus/*" in resources
    assert "arn:aws:s3:::finvec-corpus" in resources
    assert all("finvec-corpus" in r for r in resources)


def test_object_acls_stay_blocked_so_the_policy_is_the_only_public_route():
    # Policy-based public access only: an object ACL must not be able to expose
    # anything, even by accident.
    assert PUBLIC_ACCESS_BLOCK["BlockPublicAcls"] is True
    assert PUBLIC_ACCESS_BLOCK["IgnorePublicAcls"] is True
    # These two must be off, or the bucket policy itself would be rejected.
    assert PUBLIC_ACCESS_BLOCK["BlockPublicPolicy"] is False
    assert PUBLIC_ACCESS_BLOCK["RestrictPublicBuckets"] is False
