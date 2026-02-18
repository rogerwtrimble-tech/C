"""Secure file handling with encryption and secure deletion."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import Config


class SecureFileHandler:
    """Handle encrypted files and secure deletion."""
    
    def __init__(self):
        self._fernet: Optional[Fernet] = None
        self.deletion_passes = Config.SECURE_DELETION_PASSES
    
    def _get_fernet(self) -> Fernet:
        """Get or create Fernet instance from encryption key."""
        if self._fernet is None:
            key = Config.get_encryption_key()
            if key:
                # Derive Fernet key from AES-256 key
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"medical_extraction_salt",  # Static salt for deterministic key derivation
                    iterations=100000,
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(key))
                self._fernet = Fernet(derived_key)
            else:
                raise ValueError("Encryption key not configured")
        return self._fernet
    
    def encrypt_file(self, source_path: Path, dest_path: Optional[Path] = None) -> Path:
        """
        Encrypt a file.
        
        Args:
            source_path: Path to file to encrypt
            dest_path: Destination path (optional, defaults to source + .enc)
            
        Returns:
            Path to encrypted file
        """
        fernet = self._get_fernet()
        
        if dest_path is None:
            dest_path = source_path.with_suffix(source_path.suffix + ".enc")
        
        with open(source_path, "rb") as f:
            data = f.read()
        
        encrypted_data = fernet.encrypt(data)
        
        with open(dest_path, "wb") as f:
            f.write(encrypted_data)
        
        return dest_path
    
    def decrypt_file(self, source_path: Path, dest_path: Optional[Path] = None) -> Path:
        """
        Decrypt a file.
        
        Args:
            source_path: Path to encrypted file
            dest_path: Destination path (optional, defaults to removing .enc suffix)
            
        Returns:
            Path to decrypted file
        """
        fernet = self._get_fernet()
        
        if dest_path is None:
            if source_path.suffix == ".enc":
                dest_path = source_path.with_suffix("")
            else:
                dest_path = source_path.with_suffix(".decrypted")
        
        with open(source_path, "rb") as f:
            encrypted_data = f.read()
        
        decrypted_data = fernet.decrypt(encrypted_data)
        
        with open(dest_path, "wb") as f:
            f.write(decrypted_data)
        
        return dest_path
    
    def secure_delete(self, file_path: Path) -> bool:
        """
        Securely delete a file with multi-pass overwrite.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful
        """
        if not file_path.exists():
            return True
        
        file_size = file_path.stat().st_size
        
        try:
            # Multi-pass overwrite
            with open(file_path, "r+b") as f:
                for _ in range(self.deletion_passes):
                    # Overwrite with random data
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
                
                # Final overwrite with zeros
                f.seek(0)
                f.write(b"\x00" * file_size)
                f.flush()
                os.fsync(f.fileno())
            
            # Delete the file
            file_path.unlink()
            return True
            
        except Exception:
            # Fallback to regular deletion if secure deletion fails
            try:
                file_path.unlink()
            except Exception:
                pass
            return False
    
    def secure_delete_directory(self, dir_path: Path) -> bool:
        """
        Securely delete all files in a directory.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            True if all files deleted successfully
        """
        if not dir_path.exists():
            return True
        
        success = True
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                if not self.secure_delete(file_path):
                    success = False
        
        return success
    
    def create_temp_file(self, suffix: str = ".tmp") -> Path:
        """Create a temporary file path."""
        temp_dir = Path(tempfile.gettempdir()) / "medical_processing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        fd, path = tempfile.mkstemp(suffix=suffix, dir=str(temp_dir))
        os.close(fd)
        
        return Path(path)
    
    def create_temp_directory(self) -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.gettempdir()) / "medical_processing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        return Path(tempfile.mkdtemp(dir=str(temp_dir)))
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum matches expected value."""
        actual = self.calculate_checksum(file_path)
        return actual == expected_checksum
