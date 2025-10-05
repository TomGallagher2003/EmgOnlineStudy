"""Functions for handling TCP socket connections and data transfer.

This module defines a simple `SocketHandler` class that wraps basic
TCP client functionality: connecting, sending, receiving, and closing
a socket. It also includes retries for connection attempts and utilities
like flushing the socket buffer.
"""

import socket
import time
import select


class SocketHandler:
    """Lightweight wrapper around a TCP socket connection."""

    def __init__(self, ip: str, port: int):
        """
        Initialize a socket handler for a given endpoint.

        Args:
            ip (str): The IP address of the server to connect to.
            port (int): The TCP port number of the server.
        """
        self.ip = ip
        self.port = port
        self.socket: socket.socket | None = None

    def connect(self, retries: int = 5, retry_delay: int = 2) -> bool:
        """Establish a socket connection to the configured endpoint."""
        for i in range(retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.connect((self.ip, self.port))
                print("Connected to Socket!")
                self.socket.settimeout(20)
                return True
            except socket.error as msg:
                print(f"[connect attempt {i+1}] {msg}")
                time.sleep(retry_delay)
        raise ConnectionError("Failed to connect to socket after multiple retries")

    def close(self) -> bool:
        """Shut down and close the socket connection."""
        try:
            if self.socket:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
                print("Socket Closed!")
            return True
        except socket.error as msg:
            print(msg)
            return False

    def send(self, data: bytes):
        """Send data over the socket."""
        try:
            if not self.socket:
                raise RuntimeError("Socket not connected")
            self.socket.sendall(data)
            self.socket.settimeout(20)
            return True
        except socket.error as msg:
            print(msg)
            return None

    def receive(self, size: int) -> bytes | None:
        """Receive data from the socket."""
        try:
            if not self.socket:
                raise RuntimeError("Socket not connected")
            return self.socket.recv(size)
        except socket.error as msg:
            print(msg)
            return None

    def flush(self, max_ms: int = 100) -> int:
        """Flush any residual data in the socket buffer.

        Non-blocking read loop clears pending bytes until empty or until
        the max time budget is exceeded.

        Args:
            max_ms (int): Maximum time in milliseconds to spend flushing.

        Returns:
            int: Total number of bytes discarded.
        """
        if not self.socket:
            return 0

        total = 0
        end_time = time.time() + (max_ms / 1000.0)

        # temporarily set non-blocking
        self.socket.setblocking(False)
        try:
            while time.time() < end_time:
                r, _, _ = select.select([self.socket], [], [], 0)
                if not r:
                    break
                try:
                    chunk = self.socket.recv(4096)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                total += len(chunk)
        finally:
            # restore blocking mode
            self.socket.setblocking(True)

        if total:
            print(f"[flush] discarded {total} bytes from socket buffer")
        return total
