# beatmania-aidx / chartGenerator

`chartGenerator.py` は、音声ファイル（WAV形式）を解析し、音楽ゲーム用の譜面データ（JSON形式）を自動生成するPythonスクリプトです。
LibROSAを用いて楽曲のメロディ（調波成分）やパーカッション（打楽器成分）のリズム、オンセット（音の立ち上がり）、周波数帯域ごとの強度などを検出し、楽曲の展開（Drop, Build-up, Breakなど）に合わせた譜面を生成します。

## 実行環境の準備

このプロジェクトを実行するには、Pythonといくつかのライブラリが必要です。

```bash
pip install librosa numpy scipy
```

また、ブラウザでゲームをプレイ（または表示）するためには、ローカルサーバーを立ち上げる必要があります。
ローカルサーバーの起動には以下のコマンドを実行します。

```bash
python server.py
```
デフォルトでは `http://localhost:8000` でサーバーが起動します。
ポート番号を変更したい場合は `python server.py 8080` のようにポート番号を引数に渡してください。

## 譜面生成スクリプトの実行方法

基本的には以下のコマンド形式で実行します。

```bash
python chartGenerator.py [入力ファイル名] [譜面タイプ] [目標ノーツ数]
```

### 引数の詳細

1.  **入力ファイル名（省略可）**:
    *   対象とする音声ファイル名（拡張子 `.wav` は省略可能）を指定します。
    *   デフォルトは `"song"` (`song.wav` を探します)。
    *   例: `audio` と指定すると `audio.wav` を読み込みます。
2.  **譜面タイプ（省略可）**:
    *   出力ファイル名に含まれる譜面の種類（難易度など）を指定します。
    *   デフォルトは `"normal"` です。
3.  **目標ノーツ数（省略可）**:
    *   生成する譜面の目標ノーツ数を指定します。
    *   デフォルトは `800` です。

### 実行例

**例1: デフォルト設定で実行する**
ディレクトリ内にある `song.wav` を読み込み、目標ノーツ数 800 で譜面（タイプ: normal）を生成します。
```bash
python chartGenerator.py
```
出力先: `song_normal.chart.json`

**例2: ファイル名と譜面タイプを指定して実行する**
ディレクトリ内にある `my_music.wav` を読み込み、目標ノーツ数 800 で譜面（タイプ: hard）を生成します。
```bash
python chartGenerator.py my_music hard
```
出力先: `my_music_hard.chart.json`

**例3: ファイル名、譜面タイプ、目標ノーツ数を指定して実行する**
ディレクトリ内にある `track01.wav` を読み込み、目標ノーツ数 1200 の譜面（タイプ: extreme）を生成します。
```bash
python chartGenerator.py track01 extreme 1200
```
出力先: `track01_extreme.chart.json`

## 出力されるファイル

スクリプトが正常に完了すると、入力ファイル名（拡張子なし）に `_` と指定した譜面タイプが追加された `.chart.json` というファイルが生成されます（例: `song_normal.chart.json`）。これが生成された譜面データとなります。
