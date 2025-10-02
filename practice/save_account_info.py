from qiskit_ibm_runtime import QiskitRuntimeService
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def load_config_from_file(path: Path):
    """Load KEY=VALUE lines from a config file. Returns a dict."""
    result = {}
    if not path.exists():
        return result
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            result[k.strip()] = v.strip()
    except Exception as e:
        logging.error("Failed to read config file %s: %s", path, e)
    return result


def main():
    # 優先順: 環境変数 > quantum-cascade/.config
    config = {}
    repo_root = Path(__file__).resolve().parents[1]
    config_file = repo_root / '.config'
    config.update(load_config_from_file(config_file))

    api_key = os.environ.get('IBM_API_KEY') or config.get('IBM_API_KEY')
    instance = os.environ.get('IBM_INSTANCE_ID') or config.get('IBM_INSTANCE_ID')

    if not api_key or not instance:
        logging.warning(
            'IBM API key or instance id not set. Set env IBM_API_KEY and IBM_INSTANCE_ID or create %s from .config.template',
            config_file
        )
        return

    # 保存してデフォルトとして設定
    try:
        QiskitRuntimeService.save_account(
            channel='ibm_cloud',
            token=api_key,
            instance=instance,
            set_as_default=True
        )
        logging.info('IBM Quantum account saved as default')
    except Exception as e:
        logging.error('Failed to save IBM Quantum account: %s', e)


    
if __name__ == '__main__':
    main()
