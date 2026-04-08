"""
ConnectRPC sync client for bitgn.harness.HarnessService
Generated locally from proto/bitgn/harness.proto (bypass buf.build)
"""
from connectrpc._client_sync import ConnectClientSync
from connectrpc.method import IdempotencyLevel, MethodInfo

from bitgn.harness_pb2 import (
    EndTrialRequest, EndTrialResponse,
    GetBenchmarkRequest, GetBenchmarkResponse,
    GetRunRequest, GetRunResponse,
    GetTrialRequest, GetTrialResponse,
    StartPlaygroundRequest, StartPlaygroundResponse,
    StartRunRequest, StartRunResponse,
    StartTrialRequest, StartTrialResponse,
    StatusRequest, StatusResponse,
    SubmitRunRequest, SubmitRunResponse,
)

_SVC = "bitgn.harness.HarnessService"
_U = IdempotencyLevel.UNKNOWN


def _m(name, req, res):
    return MethodInfo(name=name, service_name=_SVC, input=req, output=res, idempotency_level=_U)


class HarnessServiceClientSync:
    def __init__(self, address: str, api_key: str = ""):
        self._c = ConnectClientSync(address)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    def _h(self):
        return self._headers

    def status(self, req: StatusRequest) -> StatusResponse:
        return self._c.execute_unary(request=req, method=_m("Status", StatusRequest, StatusResponse), headers=self._h())

    def get_benchmark(self, req: GetBenchmarkRequest) -> GetBenchmarkResponse:
        return self._c.execute_unary(request=req, method=_m("GetBenchmark", GetBenchmarkRequest, GetBenchmarkResponse), headers=self._h())

    def start_run(self, req: StartRunRequest) -> StartRunResponse:
        return self._c.execute_unary(request=req, method=_m("StartRun", StartRunRequest, StartRunResponse), headers=self._h())

    def get_run(self, req: GetRunRequest) -> GetRunResponse:
        return self._c.execute_unary(request=req, method=_m("GetRun", GetRunRequest, GetRunResponse), headers=self._h())

    def submit_run(self, req: SubmitRunRequest) -> SubmitRunResponse:
        return self._c.execute_unary(request=req, method=_m("SubmitRun", SubmitRunRequest, SubmitRunResponse), headers=self._h())

    def start_playground(self, req: StartPlaygroundRequest) -> StartPlaygroundResponse:
        return self._c.execute_unary(request=req, method=_m("StartPlayground", StartPlaygroundRequest, StartPlaygroundResponse), headers=self._h())

    def start_trial(self, req: StartTrialRequest) -> StartTrialResponse:
        return self._c.execute_unary(request=req, method=_m("StartTrial", StartTrialRequest, StartTrialResponse), headers=self._h())

    def get_trial(self, req: GetTrialRequest) -> GetTrialResponse:
        return self._c.execute_unary(request=req, method=_m("GetTrial", GetTrialRequest, GetTrialResponse), headers=self._h())

    def end_trial(self, req: EndTrialRequest) -> EndTrialResponse:
        return self._c.execute_unary(request=req, method=_m("EndTrial", EndTrialRequest, EndTrialResponse), headers=self._h())
