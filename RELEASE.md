# Release procedure

このリポジトリを **GitHub に push → CI で Windows wheel 含む全 wheel をビルド → TestPyPI → 本番 PyPI** の流れで公開する手順をまとめたものです。初回 (= リポジトリの git 化と PyPI 名の確保) はそこそこ作業がありますが、2 回目以降は **tag を打つだけ** で済みます。

---

## 全体像

```
[ローカル svn]
  ↓ git 化 (1 回だけ)
[ローカル git] ── push ──→ [GitHub] ──→ Actions (cibuildwheel)
                                            ├─ Linux / macOS wheel (必須)
                                            ├─ Windows wheel (best-effort)
                                            ├─ sdist
                                            └─ cpp_cli_compat
                                            ↓
                                            ├─ workflow_dispatch ─→ TestPyPI
                                            └─ release published ─→ PyPI
```

GitHub Actions は OIDC を使った **Trusted Publishing** で PyPI にアップロードします。API トークンをリポジトリに置く必要はありません。

---

## Phase 1: ローカルを git 化

upstream の `.svn/` は 2011 年の snapshot で、ローカルに保持すべき svn 変更はない前提です。

```bash
cd /Users/yt/Prog/10_cpp/sketchsort-0.0.8

# 安全のため、念のため .svn 系を tar に退避 (不要なら skip)
tar czf /tmp/sketchsort-svn-backup.tar.gz \
    .svn .lock-wscript .waf-1.5.11-* src/.svn src/build 2>/dev/null || true

# git 初期化
git init -b main

# .gitignore は既に整備済み (.svn/, .waf-*/, build/, *.o 等)
# add の前に確認:
git status --short | head -30          # ← .svn が含まれないことを確認
git check-ignore -v .svn src/.svn      # ← どちらも .gitignore で無視される旨を表示

# 全部ステージング
git add -A

# 念のため Boost を含めて何件入るかカウント
git diff --cached --stat | tail -1

git commit -m "Initial commit: SketchSort 0.1.0 Python packaging

Repackages upstream SketchSort 0.0.8 (Yasuo Tabei, 2011) as a pip-installable
Python package with pybind11 bindings, CMake/scikit-build-core build, and a
CI wheel pipeline (cibuildwheel).

See README.md and RELEASE.md for usage and release procedure."
```

`.svn` 系が漏れたら:

```bash
git rm -rf --cached .svn src/.svn src/build
echo ".svn/" >> .gitignore  # 既に書いてあるはず
git commit --amend --no-edit
```

---

## Phase 2: GitHub にリポジトリを作って push

### 2-1. リポジトリ作成

`gh` CLI を使う場合 (推奨):

```bash
gh auth login                  # 初回のみ
gh repo create <your-account>/sketchsort \
    --public \
    --description "Fast all-pairs cosine-similarity search via random projection sketches" \
    --source . \
    --remote origin \
    --push
```

UI でやる場合:

1. https://github.com/new で **空のリポジトリ** を作る (README/LICENSE/.gitignore のチェックは外す)
2. 表示される URL でローカルから push:
   ```bash
   git remote add origin git@github.com:<your-account>/sketchsort.git
   git branch -M main
   git push -u origin main
   ```

### 2-2. CI を初回起動 (Windows wheel の確認用)

push しただけでは Actions は走りません (`pull_request` / `release` / `workflow_dispatch` のみ trigger)。手動で発火:

```bash
gh workflow run wheels.yml
gh run watch     # 進行状況を tail
```

UI なら: Actions タブ → "Build wheels" → "Run workflow" ボタン。

完了後の確認ポイント:

- `wheel ubuntu-latest` `wheel macos-13` `wheel macos-14` — **すべて成功必須**
- `wheel windows-latest` — `continue-on-error: true` なので失敗しても全体は ✓ になる。失敗ログは `gh run view --log-failed` で取得
- `sdist` — 成功必須
- `cpp_cli_compat` — 成功必須 (これが落ちると Python と C++ CLI の output が乖離している証拠)

### 2-3. (オプション) Windows を直す or 一旦無視

Windows が落ちる場合は、原因 (たいてい vendored Boost の MSVC 非互換) を `gh run view --log-failed` で確認します。当面 Windows を諦めるなら matrix から外す:

```yaml
# .github/workflows/wheels.yml の build_wheels.matrix.include から
# - { os: windows-latest, allow-failure: true }
# の行を削除して push
```

---

## Phase 3: PyPI で Trusted Publishing を設定

**先に PyPI 側で「publisher」を登録**しておかないと、後で Actions が認証エラーで落ちます。

### 3-1. PyPI アカウントと "pending publisher"

1. https://pypi.org/ にアカウント作成 (or login) + 2FA 有効化
2. https://pypi.org/manage/account/publishing/ にアクセス
3. "Add a new pending publisher" で:

   | 項目 | 値 |
   |---|---|
   | PyPI Project Name | `sketchsort` |
   | Owner | `<your-github-account>` |
   | Repository name | `sketchsort` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` |

4. 同じ流れで **TestPyPI** にも登録:
   - https://test.pypi.org/manage/account/publishing/
   - Environment name は `testpypi`

### 3-2. GitHub 側で environment を作成

GitHub リポジトリの Settings → Environments → New environment:

- 名前 `pypi`
  - (任意) "Required reviewers" を自分にしておくと、公開前にワンクリック承認のステップが入って事故防止になる
- 名前 `testpypi`
  - (任意) protection rule なし、自由に dry-run できる

これで Trusted Publishing 経路が完成。トークンの管理は不要です。

---

## Phase 4: TestPyPI で dry-run

本番 PyPI に上げる前に必ず一度 TestPyPI で試します。

### 4-1. ワークフローを workflow_dispatch で起動

```bash
gh workflow run wheels.yml
gh run watch
```

`wheels.yml` の `publish_testpypi` job が **workflow_dispatch トリガー時のみ動く** 設定なので、これで TestPyPI に publish されます。

### 4-2. 別環境で動作確認

```bash
python -m venv /tmp/ss_test
/tmp/ss_test/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    sketchsort
# --extra-index-url を入れる理由: TestPyPI には numpy 本体が無いので prod を併用

/tmp/ss_test/bin/python -c "
import numpy as np, sketchsort
print('version:', sketchsort.__version__)
X = np.random.default_rng(0).normal(size=(100, 16)).astype(np.float32)
print(sketchsort.search(X, cos_dist=0.5, seed=42).shape)
"
```

OS が異なる別マシン (Linux / Windows VM) でも同じことができれば、その OS の wheel が壊れていないことを示せる。

### 4-3. うまくいかなかったら version を上げて再アップロード

TestPyPI / PyPI とも、**同じ version の再アップロードは原則禁止**です (`skip-existing: true` で wheel スキップはできるが、修正後の同 version 再 publish は不可)。

TestPyPI で問題が見つかったら:

1. `pyproject.toml` / `python/sketchsort/__init__.py` / `src/Main.cpp` / `README.md` の version を `0.1.0` → `0.1.0.dev1` 等に上げる
2. commit + push
3. `gh workflow run wheels.yml` で再 dry-run

---

## Phase 5: 本番 PyPI へリリース

TestPyPI で問題なければ本番。**GitHub Release を作ると自動で publish される**設定なので、ローカルで `twine upload` を打つ必要はありません。

### 5-1. version を確定

`0.1.0.dev*` のような pre-release を経て決めた version を `0.1.0` (or `0.1.1` 等) に戻す:

```bash
# 該当箇所:
# pyproject.toml          version = "0.1.0"
# python/sketchsort/__init__.py  __version__ = "0.1.0"
# src/Main.cpp            "SketchSort version 0.1.0"
# README.md               (タイトル / changelog)
git diff       # 確認
git add -A
git commit -m "Release 0.1.0"
git push
```

### 5-2. tag を打って Release を発行

```bash
git tag v0.1.0
git push origin v0.1.0

gh release create v0.1.0 \
    --title "v0.1.0 — first PyPI release" \
    --notes-file - <<'EOF'
First Python-packaged release of SketchSort, based on upstream 0.0.8 (2011).

## Highlights
- `sketchsort.search(X, ...)` NumPy API
- `sketchsort.run_from_file(...)` byte-exact CLI parity
- `sketchsort` console script
- Deterministic by default (`-seed 0`)

See README.md for full details and CHANGELOG.
EOF
```

UI から発行する場合: Releases → "Draft a new release" → tag `v0.1.0` を選ぶ → Publish.

### 5-3. CI を見守る

`gh run watch` で `publish_pypi` job まで完走するのを確認。GitHub Environments の `pypi` に protection rule (Required reviewers) を入れていたら、Actions の画面で承認ボタンを押す必要があります。

### 5-4. 一般ユーザの動作確認

```bash
python -m venv /tmp/ss_prod && /tmp/ss_prod/bin/pip install sketchsort
/tmp/ss_prod/bin/python -c "import sketchsort; print(sketchsort.__version__)"
```

---

## 2 回目以降 (= 通常のリリース)

bug fix release 等は以下だけ:

```bash
# 1. version をすべての場所で上げる (0.1.0 → 0.1.1)
# 2. commit + push
git commit -am "Release 0.1.1"
git push

# 3. tag + release で自動 publish
git tag v0.1.1
git push origin v0.1.1
gh release create v0.1.1 --title "v0.1.1" --generate-notes
```

これで CI が wheel をビルドし、PyPI に公開してくれます。

---

## トラブルシュート

| 症状 | 原因 / 対処 |
|---|---|
| `publish_pypi` が `403 invalid-publisher` で落ちる | PyPI 側の Trusted Publisher 設定の `Workflow name` / `Environment` / repo owner が一致していない |
| `pypi-publish` action が `version already exists` で skip | 既存 version 番号への上書きは不可。version を bump して push |
| Windows wheel だけ赤いまま | `continue-on-error: true` で release ジョブは止まらない。`CIBW_SKIP` に該当 cp を足すか、原因 (Boost MSVC 互換性) を直す |
| `cpp_cli_compat` が落ちる | C++ CLI と Python の `cos_dist` text 表現が乖離。`-DCMAKE_BUILD_TYPE=Release` が効いているか確認。`-ffast-math` が混入していないか |
| TestPyPI で `numpy` が見つからない | `pip install --extra-index-url https://pypi.org/simple/ sketchsort` で prod から numpy を取らせる |
| `gh` コマンドが存在しない | `brew install gh` |

---

## 参考リンク

- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- cibuildwheel: https://cibuildwheel.pypa.io/
- pypa/gh-action-pypi-publish: https://github.com/pypa/gh-action-pypi-publish
- GitHub Environments: https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
