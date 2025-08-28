import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import secrets
import random
import oqs  # Open Quantum Safe for PQ crypto

# Deep Learning Models
class EncryptionModel(nn.Module):
    def __init__(self):
        super(EncryptionModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input: size, sensitivity
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Output: 3 KEMs
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class AdversarialDecryptor(nn.Module):
    def __init__(self):
        super(AdversarialDecryptor, self).__init__()
        self.fc1 = nn.Linear(32, 128)  # Input: ciphertext block
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: vulnerability prob
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class QCCHEE:
    def __init__(self):
        # Initialize DL models (simulated pre-trained weights)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encryption_model = EncryptionModel().to(self.device)
        self.adversarial_decryptor = AdversarialDecryptor().to(self.device)
        # Initialize with random weights (in real use, load pre-trained)
        self.encryption_model.eval()
        self.adversarial_decryptor.eval()
        
        # PQ algorithms
        self.kems = ['KYBER512', 'FRODOKEM', 'BIKE']  # liboqs names
        self.signatures = ['DILITHIUM2', 'FALCON', 'SPHINCSPLUS']
        self.failed_attempts = 0
        self.max_attempts = 5

    def lorenz_attractor(self, state, t, sigma=10, beta=8/3, rho=28):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    def generate_lorenz_entropy(self, size):
        t = np.linspace(0, 10, size * 10)
        seed = secrets.token_bytes(24)
        initial_state = [float.fromhex(seed[i:i+8].hex()) % 1.0 for i in (0, 8, 16)]
        solution = odeint(self.lorenz_attractor, initial_state, t)
        return bytes([(int((x * 100) % 256)) for x in solution[:, 0]][:size])

    def logistic_map(self, r=3.99, x=None, n=256):
        x = x or (secrets.randbits(32) / (1 << 32))
        result = []
        for _ in range(n):
            x = r * x * (1 - x)
            result.append(int(x * 256) % 256)
        return result

    def chebyshev_map(self, x=None, n=256):
        x = x or (secrets.randbits(32) / (1 << 32))
        result = []
        for _ in range(n):
            x = math.cos(5 * math.acos(x))
            result.append(int((x + 1) * 128) % 256)
        return result

    def tent_map(self, x=None, n=256):
        x = x or (secrets.randbits(32) / (1 << 32))
        result = []
        for _ in range(n):
            x = 2 * x if x < 0.5 else 2 * (1 - x)
            result.append(int(x * 256) % 256)
        return result

    def henon_map(self, x=None, y=None, a=1.4, b=0.3, n=256):
        x = x or (secrets.randbits(32) / (1 << 32))
        y = y or (secrets.randbits(32) / (1 << 32))
        result = []
        for _ in range(n):
            x_new = y + 1 - a * x**2
            y = b * x
            x = x_new
            result.append(int(x * 256) % 256)
        return result

    def generate_chaos_entropy(self, size):
        lorenz = self.generate_lorenz_entropy(size // 5)
        log = self.logistic_map(n=size // 5)
        cheb = self.chebyshev_map(n=size // 5)
        tent = self.tent_map(n=size // 5)
        hen = self.henon_map(n=size // 5)
        pool = list(lorenz) + log + cheb + tent + hen
        random.seed(secrets.randbits(128))
        random.shuffle(pool)
        return bytes(pool[:size])

    def triage_kem(self, plaintext: bytes) -> str:
        size = len(plaintext)
        sensitivity = sum(plaintext) % 1000
        inputs = torch.tensor([[size / 1024, sensitivity / 1000]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.encryption_model(inputs)
        return self.kems[torch.argmax(probs, dim=1).item()]

    def check_vulnerability(self, ciphertext: bytes) -> bool:
        block = torch.tensor([list(ciphertext[:32]) + [0] * (32 - min(32, len(ciphertext)))], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            vuln_prob = self.adversarial_decryptor(block).item()
        return vuln_prob > 0.9

    def encrypt(self, plaintext: bytes) -> bytes:
        if self.failed_attempts >= self.max_attempts:
            print("Key self-destruct activated!")
            return b""  # Self-destruct

        # Step 1: Triage
        kem_name = self.triage_kem(plaintext)
        signature_name = random.choice(self.signatures)

        # Step 2: PQ Key Encapsulation
        kem = oqs.KeyEncapsulation(kem_name if kem_name == 'KYBER512' else 'KYBER512')  # Fallback
        public_key = kem.generate_keypair()
        ciphertext_kem, shared_secret = kem.encap_secret(public_key)

        # Step 3: Chaos entropy
        size = len(plaintext)
        entropy = self.generate_chaos_entropy(size)

        # Step 4: Derive stream key
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'qc-chee')
        stream_key = hkdf.derive(shared_secret + entropy)

        # Step 5: Generate chaotic stream
        stream = bytearray()
        prev_block = stream_key
        block_size = 32
        for i in range(0, size, block_size):
            block = hashlib.sha3_256(prev_block).digest()
            stream += block
            prev_block = block + plaintext[i:i+block_size]

        stream = stream[:size]

        # Step 6: XOR and permute
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
        permuted = bytearray(ciphertext)
        for i in range(len(permuted)):
            permuted[i] = (permuted[i] + i) % 256

        # Step 7: Adversarial check
        if self.check_vulnerability(bytes(permuted)):
            print("Vulnerability detected; adapting...")
            entropy = self.generate_chaos_entropy(size)
            stream_key = hkdf.derive(shared_secret + entropy)
            stream = bytearray()
            prev_block = stream_key
            for i in range(0, size, block_size):
                block = hashlib.sha3_256(prev_block).digest()
                stream += block
                prev_block = block + plaintext[i:i+block_size]
            stream = stream[:size]
            ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
            permuted = bytearray(ciphertext)
            for i in range(len(permuted)):
                permuted[i] = (permuted[i] + i) % 256

        # Step 8: Sign
        sig = oqs.Signature('DILITHIUM2' if signature_name == 'DILITHIUM2' else 'DILITHIUM2')
        signer_public_key = sig.generate_keypair()
        signature = sig.sign(bytes(permuted))

        return bytes(permuted) + signature

    def decrypt(self, ciphertext: bytes, kem_secret_key: bytes) -> bytes:
        self.failed_attempts += 1
        if self.failed_attempts >= self.max_attempts:
            print("Key self-destruct activated!")
            return b""

        # Extract signature
        sig_size = 2420  # Dilithium2 signature size
        signature = ciphertext[-sig_size:]
        ciphertext = ciphertext[:-sig_size]

        # Verify signature
        sig = oqs.Signature('DILITHIUM2')
        if not sig.verify(ciphertext, signature, sig.public_key):
            raise ValueError("Signature verification failed")

        # Decapsulate shared secret
        kem = oqs.KeyEncapsulation('KYBER512')
        shared_secret = kem.decap_secret(ciphertext[:kem.length_ciphertext], kem_secret_key)
        ciphertext = ciphertext[kem.length_ciphertext:]

        # Regenerate stream
        size = len(ciphertext)
        entropy = self.generate_chaos_entropy(size)
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'qc-chee')
        stream_key = hkdf.derive(shared_secret + entropy)

        stream = bytearray()
        prev_block = stream_key
        block_size = 32
        for i in range(0, size, block_size):
            block = hashlib.sha3_256(prev_block).digest()
            stream += block
            prev_block = block + ciphertext[i:i+block_size]
        stream = stream[:size]

        # Reverse permutation
        permuted = bytearray(ciphertext)
        for i in range(len(permuted)):
            permuted[i] = (permuted[i] - i) % 256

        # Decrypt
        plaintext = bytes(a ^ b for a, b in zip(permuted, stream))
        return plaintext

    @staticmethod
    def visualize_lorenz():
        t = np.linspace(0, 10, 1000)
        initial_state = [1.0, 1.0, 1.0]
        solution = odeint(QCCHEE().lorenz_attractor, initial_state, t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], lw=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Lorenz Attractor')
        plt.show()

# Example usage
if __name__ == "__main__":
    engine = QCCHEE()
    data = b"Sensitive test data"
    
    # Generate PQ keypair
    kem = oqs.KeyEncapsulation('KYBER512')
    public_key = kem.generate_keypair()
    secret_key = kem.secret_key  # Store securely in production
    
    encrypted = engine.encrypt(data)
    decrypted = engine.decrypt(encrypted, secret_key)
    print(f"Original: {data}")
    print(f"Encrypted: {encrypted.hex()}")
    print(f"Decrypted: {decrypted}")
    assert data == decrypted, "Decryption failed"
    
    # Visualize Lorenz attractor
    engine.visualize_lorenz()