from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile

def main():
    # ログイン（資格情報は save_account 済みなら不要引数でOK）
    service = QiskitRuntimeService()

    # 利用可能な実機で一番空いているものを選ぶ例（シミュレータは除外）
    backend = service.least_busy(operational=True, simulator=False)

    # 簡単なベル状態回路
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0,1)
    qc.measure([0,1],[0,1])

    tc = transpile(qc, backend=backend)
    print(tc.draw())

    # SamplerV2 は "mode=backend" でそのバックエンドに投げる
    sampler = Sampler(mode=backend)

    job = sampler.run([tc], shots=1024)
    result = job.result()
    print(job.metadata)
    print(job.metrics)

    # V2 では pub_result.data.<register名>.get_counts() でカウント取得（通常は 'meas'）
    pub_result = result[0]
    counts = pub_result.join_data().get_counts()
    print(counts)

if __name__ == '__main__':
    main()