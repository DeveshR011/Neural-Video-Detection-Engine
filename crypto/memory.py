import ctypes
import os
import sys

# Platform specific memory locking
if os.name == 'nt':
    # Windows
    kernel32 = ctypes.windll.kernel32
    VirtualLock = kernel32.VirtualLock
    VirtualUnlock = kernel32.VirtualUnlock
    RtlZeroMemory = kernel32.RtlZeroMemory

    def lock_memory(address: int, size: int) -> bool:
        """Lock memory into physical RAM, preventing paging to disk."""
        return VirtualLock(ctypes.c_void_p(address), ctypes.c_size_t(size)) != 0

    def unlock_memory(address: int, size: int) -> bool:
        """Unlock memory."""
        return VirtualUnlock(ctypes.c_void_p(address), ctypes.c_size_t(size)) != 0

    def zero_memory(address: int, size: int):
        """Securely overwrite memory with zeros."""
        RtlZeroMemory(ctypes.c_void_p(address), ctypes.c_size_t(size))

else:
    # POSIX (Linux, macOS, etc.)
    libc = ctypes.CDLL(None)
    mlock = libc.mlock
    munlock = libc.munlock
    memset = libc.memset

    def lock_memory(address: int, size: int) -> bool:
        return mlock(ctypes.c_void_p(address), ctypes.c_size_t(size)) == 0

    def unlock_memory(address: int, size: int) -> bool:
        return munlock(ctypes.c_void_p(address), ctypes.c_size_t(size)) == 0

    def zero_memory(address: int, size: int):
        memset(ctypes.c_void_p(address), 0, ctypes.c_size_t(size))

class SecureBuffer:
    """A context manager for an automatic zeroing and locked memory buffer."""
    def __init__(self, size: int):
        self.size = size
        # Allocate a byte array
        self.buffer = bytearray(size)
        # Get memory address of the underlying data
        # Note: id(self.buffer) is not the address of the data buffer
        # In CPython, a bytearray data is a pointer stored at a specific offset.
        # However, for simplicity and standard python interfacing, acquiring the
        # exact buffer pointer in pure python using ctypes:
        
        Char_Array = ctypes.c_char * self.size
        # To truly control memory, allocate via ctypes
        self._raw_array = Char_Array()
        self.address = ctypes.addressof(self._raw_array)
        
        # Lock the memory
        if not lock_memory(self.address, self.size):
            print("Warning: Could not lock memory. Paging to disk is possible.", file=sys.stderr)
            
    def get_address(self) -> int:
        return self.address
        
    def get_buffer(self) -> ctypes.Array:
        return self._raw_array

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Securely zero out the memory on context exit
        zero_memory(self.address, self.size)
        # Unlock the memory
        unlock_memory(self.address, self.size)

    def write(self, data: bytes):
        if len(data) > self.size:
            raise ValueError("Data exceeds buffer size")
        ctypes.memmove(self.address, data, len(data))

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self.size
        return ctypes.string_at(self.address, size)
