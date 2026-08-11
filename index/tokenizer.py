def tokenize_cn(text: str) -> list[str]:
    # 决策2：全部空白字符删除：空格、制表、换行
    s = "".join(text.split())
    if not s:
        return []
    if len(s) == 1:
        return [s]
    tokens: list[str] = []
    max_i = len(s) - 2
    for i in range(max_i + 1):
        tokens.append(s[i:i+2])
    return tokens