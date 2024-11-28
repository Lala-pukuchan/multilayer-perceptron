# 仮想環境を作成し、依存関係をインストール
setup:
	python3 -m venv venv

activate:
	@echo "Run 'source venv/bin/activate' in your shell to activate the virtual environment."

install:
	pip install --upgrade pip
	pip install -r requirements.txt

# クリーンアップ
clean:
	rm -rf venv
	rm -rf __pycache__
