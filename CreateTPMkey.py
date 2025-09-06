fapi = FAPI()
key_created = fapi.create_key(
    path=b"/HS/SRK/new_signing_key",
    type_="sign",
    policy_path=b"/policy/my_policy",
    auth_value=b"my_password",
    exists_ok=True
)
print("Key created: {}".format(key_created))
fapi.close()  # Clean up
