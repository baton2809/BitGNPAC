"""
ConnectRPC sync client for bitgn.vm.pcm.PcmRuntime
Generated locally from proto/bitgn/vm/pcm.proto (bypass buf.build)
"""
from connectrpc._client_sync import ConnectClientSync
from connectrpc.method import IdempotencyLevel, MethodInfo

from bitgn.vm.pcm_pb2 import (
    AnswerRequest, AnswerResponse,
    ContextRequest, ContextResponse,
    DeleteRequest, DeleteResponse,
    FindRequest, FindResponse,
    ListRequest, ListResponse,
    MkDirRequest, MkDirResponse,
    MoveRequest, MoveResponse,
    ReadRequest, ReadResponse,
    SearchRequest, SearchResponse,
    TreeRequest, TreeResponse,
    WriteRequest, WriteResponse,
)

_SVC = "bitgn.vm.pcm.PcmRuntime"
_U = IdempotencyLevel.UNKNOWN


def _m(name, req, res):
    return MethodInfo(name=name, service_name=_SVC, input=req, output=res, idempotency_level=_U)


class PcmRuntimeClientSync:
    def __init__(self, address: str):
        self._c = ConnectClientSync(address)

    def read(self, req: ReadRequest) -> ReadResponse:
        return self._c.execute_unary(request=req, method=_m("Read", ReadRequest, ReadResponse))

    def write(self, req: WriteRequest) -> WriteResponse:
        return self._c.execute_unary(request=req, method=_m("Write", WriteRequest, WriteResponse))

    def delete(self, req: DeleteRequest) -> DeleteResponse:
        return self._c.execute_unary(request=req, method=_m("Delete", DeleteRequest, DeleteResponse))

    def mk_dir(self, req: MkDirRequest) -> MkDirResponse:
        return self._c.execute_unary(request=req, method=_m("MkDir", MkDirRequest, MkDirResponse))

    def move(self, req: MoveRequest) -> MoveResponse:
        return self._c.execute_unary(request=req, method=_m("Move", MoveRequest, MoveResponse))

    def list(self, req: ListRequest) -> ListResponse:
        return self._c.execute_unary(request=req, method=_m("List", ListRequest, ListResponse))

    def tree(self, req: TreeRequest) -> TreeResponse:
        return self._c.execute_unary(request=req, method=_m("Tree", TreeRequest, TreeResponse))

    def find(self, req: FindRequest) -> FindResponse:
        return self._c.execute_unary(request=req, method=_m("Find", FindRequest, FindResponse))

    def search(self, req: SearchRequest) -> SearchResponse:
        return self._c.execute_unary(request=req, method=_m("Search", SearchRequest, SearchResponse))

    def context(self, req: ContextRequest) -> ContextResponse:
        return self._c.execute_unary(request=req, method=_m("Context", ContextRequest, ContextResponse))

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        return self._c.execute_unary(request=req, method=_m("Answer", AnswerRequest, AnswerResponse))
