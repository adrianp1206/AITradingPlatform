import subprocess

tickers = ['JNJ', 'NVDA', 'JPM', 'XOM', 'NKE', 'LMT']

for ticker in tickers:
    filename = f"train_full_rl_{ticker}.py"
    print(f"\n🚀 Running: {filename}")
    try:
        subprocess.run(["python", filename], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run {filename}: {e}")
