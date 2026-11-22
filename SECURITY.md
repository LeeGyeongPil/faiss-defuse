# Security policy / 보안 정책

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

Especially welcome: a pickle (`index.pkl`) that `faiss-defuse scan` reports as `safe` but that runs code or does something unexpected when loaded.
특히 환영하는 제보: `faiss-defuse scan` 이 `safe` 로 판정했는데 불러올 때 코드를 실행하거나 예상 밖으로 동작하는 피클.

faiss-defuse does not sandbox FAISS's own file parser; see "What it does not guarantee" in the README.
faiss-defuse 는 FAISS 자체 파일 파서를 격리하지 않습니다. README 의 "보장하지 않는 것" 절을 보세요.
