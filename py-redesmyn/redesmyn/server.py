from redesmyn.py_redesmyn import Endpoint, PyServer
from typing import List, Self


class Server:
    _pyserver: PyServer
    _endpoints: List[Endpoint]

    def __init__(self) -> None:
        self._pyserver = PyServer()
        self._endpoints = []

    def register(self, endpoint: Endpoint) -> Self:
        self._endpoints.append(endpoint)
        self._pyserver.register(endpoint)
        return self
    
    def serve(self):
        self._pyserver.serve()

    def __repr__(self) -> str:
        tab = " " * 4
        repr_endpoints = "\n".join(f"{tab}{endpoint}" for endpoint in self._endpoints)
        repr = f"Endpoints\n{repr_endpoints}"
        return repr