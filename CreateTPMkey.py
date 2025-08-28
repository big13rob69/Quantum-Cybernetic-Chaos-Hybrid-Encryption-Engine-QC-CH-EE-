from tpm2_pytss import FAPI

fapi = FAPI()
key_created = fapi.create_key(
    path=b"/HS/SRK/new_signing_key",  # Path to store the key
    type_="sign",  # Key type (e.g., for signing; other options: decrypt, restricted)
    policy_path=b"/policy/my_policy",  # Optional policy for access control
    auth_value=b"my_password",  # Authentication password
    exists_ok=True  # Don't error if path exists
)
print(f"Key created: {key_created}")
fapi.close()  # Clean up