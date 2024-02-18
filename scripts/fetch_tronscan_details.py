import os
import networkx
import requests
import json
import dotenv
from tqdm import tqdm



dotenv.load_dotenv()

# https://apilist.tronscanapi.com/api/deep/account/holderToken/basicInfo/trc20/transfer
# https://docs.tronscan.org/api-endpoints/deep-analysis#obtain-account-transfer-in-and-transfer-out-fund-distributio
USDT_token_address = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"



def download_usdt_tfs(account_address: str):
    url = f"https://apilist.tronscanapi.com/api/deep/account/holderToken/basicInfo/trc20/transfer?accountAddress={account_address}&tokenAddress={USDT_token_address}"
    response = requests.get(
        url, headers={"TRON-PRO-API-KEY": os.environ["TRONSCAN_API_KEY"]}
    )
    account_usdt_transfer_counts = response.json()

    with open(f"./tronscan/account_usdt_tfs/{account_address}.json", "w") as f:
        json.dump(account_usdt_transfer_counts, f, indent=4)


def download_account_info(account_address: str):
    url = f"https://apilist.tronscanapi.com/api/accountv2?address={account_address}"
    response = requests.get(
        url, headers={"TRON-PRO-API-KEY": os.environ["TRONSCAN_API_KEY"]}
    )
    account_info = response.json()

    with open(f"./tronscan/account_basic/{account_address}.json", "w") as f:
        json.dump(account_info, f, indent=4)


from multiprocessing.dummy import Pool as ThreadPool


def task(node):
    if type(node) is not str:
        node = node.decode()
    if node.startswith("b'"):
        node = node.removeprefix("b'").removesuffix("'")

    if os.path.exists(f"./tronscan/account_usdt_tfs/{node}.json") and os.path.exists(
        f"./tronscan/account_basic/{node}.json"
    ):
        print(f"Skipped {node}")
        return

    download_usdt_tfs(node)
    download_account_info(node)

    print(f"Downloaded {node}")


if __name__ == "__main__":
    with open("./legacy/all_1hop.gexf") as f:
        G = networkx.read_gexf(f)

    nodes = list(G.nodes())
    nodes.sort(reverse=False)
    with ThreadPool(5) as pool:
        list(tqdm(pool.imap(task, nodes), total=len(nodes)))
