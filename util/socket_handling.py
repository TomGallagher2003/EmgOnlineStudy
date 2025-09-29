"""Functions for handling TCP socket connections and data transfer.

This module defines a simple `SocketHandler` class that wraps basic
TCP client functionality: connecting, sending, receiving, and closing
a socket. It also includes retries for connection attempts and utilities
like flushing the socket buffer.
"""

import socket
import time


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
        self.socket = None

    def connect(self, retries: int = 5, retry_delay: int = 2) -> bool:
        """Establish a socket connection to the configured endpoint.

        Attempts to connect multiple times with delays in between.

        Args:
            retries (int, optional): Maximum number of connection attempts.
                Defaults to 5.
            retry_delay (int, optional): Delay in seconds between retries.
                Defaults to 2.

        Returns:
            bool: True if the connection is successfully established.

        Raises:
            ConnectionError: If all connection attempts fail.
        """
        for i in range(retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.connect((self.ip, self.port))
                print("Connected to Socket!")
                self.socket.settimeout(20)
                return True
            except socket.error as msg:
                print(msg)
                time.sleep(retry_delay)
        raise ConnectionError("Failed to connect to socket after multiple retries")

    def close(self) -> bool:
        """Shut down and close the socket connection.

        Returns:
            bool: True if the socket was closed successfully,
                False otherwise.
        """
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            print("Socket Closed!")
            return True
        except socket.error as msg:
            print(msg)
            return False

    def send(self, data: bytes):
        """Send data over the socket.

        Args:
            data (bytes): Raw bytes to send over the connection.

        Returns:
            Any: Return value of `socket.sendall`, or None if an error occurred.
        """
        try:
            sent = self.socket.sendall(data)
            self.socket.settimeout(20)
            return sent
        except socket.error as msg:
            print(msg)
            return None

    def receive(self, size: int) -> bytes | None:
        """Receive data from the socket.

        Args:
            size (int): Maximum number of bytes to read from the socket.

        Returns:
            bytes | None: Received data, or None if an error occurred.
        """
        try:
            data = self.socket.recv(size)
            return data
        except socket.error as msg:
            print(msg)
            return None

    def flush(self):
        """Flush any residual data in the socket buffer.

        Reads from the socket until no more data is available.
        """
        while True:
            data = self.socket.recv(4096)
            if not data:
                break
