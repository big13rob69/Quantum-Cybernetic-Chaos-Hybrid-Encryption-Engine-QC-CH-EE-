from tpm2_pytss import FAPI

fapi = FAPI()

# Create NV storage
nv_created = fapi.create_nv(
    path=b"/NV/my_storage",  # Path to NV area
    size=32,  # Size in bytes
    type_="bitfield",  # Type (other options: counter, pcr)
    policy_path=b"/policy/my_policy",  # Optional policy
    auth_value=b"my_password",  # Authentication
    exists_ok=True
)
print(f"NV created: {nv_created}")

# Write data (e.g., a master key)
fapi.nv_write(
    path=b"/NV/my_storage",
    data=b"my_secret_master_key_here"  # Data to store (bytes)
)

fapi.close()