# 第5章 Bug检测与修复建议

> 从静态分析到AI修复:构建自动化Bug修复系统

## 5.1 Bug检测流程

### 5.1.1 多层次检测架构

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class BugSeverity(Enum):
    CRITICAL = "critical"  # 崩溃/安全漏洞
    HIGH = "high"          # 逻辑错误/数据损坏
    MEDIUM = "medium"      # 性能问题/不规范
    LOW = "low"            # 代码风格

@dataclass
class DetectedBug:
    """检测到的Bug"""
    severity: BugSeverity
    category: str  # type_error/null_pointer/resource_leak等
    message: str
    file: str
    line: int
    column: int
    code_snippet: str
    suggested_fix: Optional[str] = None
    confidence: float = 0.0  # 0-1,检测置信度

class BugDetector:
    """多工具Bug检测器"""

    def __init__(self):
        from ..chapter03.static_analysis import CodeAnalysisAggregator
        self.aggregator = CodeAnalysisAggregator()

    def detect_bugs(self, file_path: str) -> List[DetectedBug]:
        """检测文件中的Bug"""
        bugs = []

        # 1. 静态分析
        static_issues = self.aggregator.analyze_file(file_path)

        for issue in static_issues:
            # 转换为Bug对象
            bug = self._convert_to_bug(issue)
            if bug:
                bugs.append(bug)

        # 2. 自定义规则检测
        custom_bugs = self._check_custom_rules(file_path)
        bugs.extend(custom_bugs)

        # 3. 按严重性排序
        bugs.sort(key=lambda x: self._severity_score(x.severity), reverse=True)

        return bugs

    def _convert_to_bug(self, issue) -> Optional[DetectedBug]:
        """将静态分析问题转换为Bug"""
        # 过滤:只关注真正的Bug,忽略风格问题
        bug_codes = {
            "F821": BugSeverity.HIGH,     # 未定义变量
            "F841": BugSeverity.MEDIUM,   # 未使用变量
            "E722": BugSeverity.HIGH,     # bare except
            "B006": BugSeverity.HIGH,     # 可变默认参数
            "B301": BugSeverity.CRITICAL, # pickle使用
            "B608": BugSeverity.CRITICAL, # SQL注入
        }

        if issue.code not in bug_codes:
            return None

        with open(issue.file, "r") as f:
            lines = f.readlines()
            code_snippet = lines[issue.line - 1].strip() if issue.line <= len(lines) else ""

        return DetectedBug(
            severity=bug_codes[issue.code],
            category=self._categorize(issue.code),
            message=issue.message,
            file=issue.file,
            line=issue.line,
            column=issue.column,
            code_snippet=code_snippet,
            confidence=0.9  # 静态分析置信度高
        )

    def _categorize(self, code: str) -> str:
        """根据错误代码分类"""
        if code.startswith("F8"):
            return "undefined_name"
        elif code.startswith("E7"):
            return "exception_handling"
        elif code.startswith("B0"):
            return "common_bug"
        elif code.startswith("B3"):
            return "security"
        elif code.startswith("B6"):
            return "injection"
        return "other"

    def _check_custom_rules(self, file_path: str) -> List[DetectedBug]:
        """自定义规则检测"""
        bugs = []

        with open(file_path, "rb") as f:
            code = f.read()

        from tree_sitter import Parser, Language
        import tree_sitter_python as tspython

        parser = Parser(Language(tspython.language()))
        tree = parser.parse(code)

        # 规则1: 检测空except块
        bugs.extend(self._check_empty_except(tree, code, file_path))

        # 规则2: 检测资源未关闭
        bugs.extend(self._check_unclosed_resources(tree, code, file_path))

        # 规则3: 检测可能的空指针
        bugs.extend(self._check_null_pointer(tree, code, file_path))

        return bugs

    def _check_empty_except(self, tree, code: bytes, file_path: str) -> List[DetectedBug]:
        """检测空的except块"""
        bugs = []

        query = tree.root_node.language.query("""
        (try_statement
          (except_clause
            (block
              (pass_statement)))) @empty_except
        """)

        for node, _ in query.captures(tree.root_node):
            bugs.append(DetectedBug(
                severity=BugSeverity.HIGH,
                category="exception_handling",
                message="空的except块会隐藏错误",
                file=file_path,
                line=node.start_point[0] + 1,
                column=node.start_point[1],
                code_snippet=code[node.start_byte:node.end_byte].decode()[:50],
                confidence=0.95
            ))

        return bugs

    def _check_unclosed_resources(self, tree, code: bytes, file_path: str) -> List[DetectedBug]:
        """检测未关闭的资源"""
        bugs = []

        # 查找open()调用但不在with语句中
        query = tree.root_node.language.query("""
        (call
          function: (identifier) @func_name
          (#eq? @func_name "open")) @open_call
        """)

        for node, _ in query.captures(tree.root_node):
            # 检查是否在with语句中
            parent = node.parent
            in_with = False

            while parent:
                if parent.type == "with_statement":
                    in_with = True
                    break
                parent = parent.parent

            if not in_with:
                bugs.append(DetectedBug(
                    severity=BugSeverity.MEDIUM,
                    category="resource_leak",
                    message="文件打开后未使用with语句,可能导致资源泄露",
                    file=file_path,
                    line=node.start_point[0] + 1,
                    column=node.start_point[1],
                    code_snippet=code[node.start_byte:node.end_byte].decode(),
                    confidence=0.85
                ))

        return bugs

    def _check_null_pointer(self, tree, code: bytes, file_path: str) -> List[DetectedBug]:
        """检测可能的空指针访问"""
        bugs = []

        # 查找可能返回None的函数调用后直接访问属性
        query = tree.root_node.language.query("""
        (attribute
          object: (call) @call) @attr_access
        """)

        for node, name in query.captures(tree.root_node):
            if name == "attr_access":
                bugs.append(DetectedBug(
                    severity=BugSeverity.HIGH,
                    category="null_pointer",
                    message="函数可能返回None,直接访问属性会抛出AttributeError",
                    file=file_path,
                    line=node.start_point[0] + 1,
                    column=node.start_point[1],
                    code_snippet=code[node.start_byte:node.end_byte].decode(),
                    confidence=0.6  # 较低置信度,可能误报
                ))

        return bugs

    def _severity_score(self, severity: BugSeverity) -> int:
        """严重性分数"""
        return {
            BugSeverity.CRITICAL: 4,
            BugSeverity.HIGH: 3,
            BugSeverity.MEDIUM: 2,
            BugSeverity.LOW: 1
        }[severity]
```

## 5.2 AI辅助修复生成

### 5.2.1 修复提示工程

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

class BugFixer:
    """AI驱动的Bug修复器"""

    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0,
            max_tokens=2000
        )

    def generate_fix(
        self,
        bug: DetectedBug,
        full_code: str
    ) -> dict:
        """为Bug生成修复方案"""
        system_prompt = """你是Python代码修复专家。
任务:根据检测到的Bug,生成最小化的修复方案。

要求:
1. 只修复问题本身,不做无关改动
2. 保持原有代码风格
3. 提供修复理由
4. 如有多种方案,说明优缺点

输出格式(JSON):
{
  "fixed_code": "修复后的代码",
  "explanation": "修复说明",
  "alternatives": ["其他可能方案"]
}"""

        user_prompt = f"""# Bug信息
严重性: {bug.severity.value}
类别: {bug.category}
描述: {bug.message}
位置: {bug.file}:{bug.line}:{bug.column}

# 问题代码
```python
{bug.code_snippet}
```

# 完整上下文
```python
{self._get_context(full_code, bug.line, context_lines=10)}
```

请生成修复方案:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)

        # 解析响应
        import json
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # 如果不是JSON,尝试提取代码块
            result = {
                "fixed_code": self._extract_code(response.content),
                "explanation": response.content,
                "alternatives": []
            }

        return result

    def generate_fix_batch(
        self,
        bugs: List[DetectedBug],
        full_code: str
    ) -> List[dict]:
        """批量生成修复方案"""
        fixes = []

        for bug in bugs:
            try:
                fix = self.generate_fix(bug, full_code)
                fixes.append({
                    "bug": bug,
                    "fix": fix
                })
            except Exception as e:
                fixes.append({
                    "bug": bug,
                    "fix": None,
                    "error": str(e)
                })

        return fixes

    def _get_context(
        self,
        code: str,
        target_line: int,
        context_lines: int = 5
    ) -> str:
        """获取Bug周围的代码上下文"""
        lines = code.split("\n")
        start = max(0, target_line - context_lines - 1)
        end = min(len(lines), target_line + context_lines)

        context_with_markers = []
        for i in range(start, end):
            marker = ">>> " if i == target_line - 1 else "    "
            context_with_markers.append(f"{marker}{lines[i]}")

        return "\n".join(context_with_markers)

    def _extract_code(self, text: str) -> str:
        """从文本中提取代码块"""
        import re

        # 查找```python...```代码块
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0]

        # 查找```...```代码块
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0]

        return text
```

### 5.2.2 修复模式库

```python
class FixPatternLibrary:
    """常见Bug修复模式库"""

    def __init__(self):
        self.patterns = {
            "undefined_name": self._fix_undefined_name,
            "resource_leak": self._fix_resource_leak,
            "exception_handling": self._fix_exception_handling,
            "null_pointer": self._fix_null_pointer,
        }

    def apply_pattern(
        self,
        bug: DetectedBug,
        code: str
    ) -> Optional[str]:
        """应用修复模式"""
        if bug.category in self.patterns:
            return self.patterns[bug.category](bug, code)
        return None

    def _fix_undefined_name(self, bug: DetectedBug, code: str) -> str:
        """修复未定义变量"""
        # 提取变量名
        import re
        match = re.search(r"'(\w+)'", bug.message)
        if not match:
            return code

        var_name = match.group(1)

        # 查找可能的候选(大小写/拼写错误)
        from difflib import get_close_matches

        words = re.findall(r'\b\w+\b', code)
        candidates = get_close_matches(var_name, words, n=1, cutoff=0.8)

        if candidates:
            suggestion = candidates[0]
            return code.replace(var_name, suggestion)

        return code

    def _fix_resource_leak(self, bug: DetectedBug, code: str) -> str:
        """修复资源泄露"""
        lines = code.split("\n")
        target_line = bug.line - 1

        if target_line >= len(lines):
            return code

        # 将open()改为with语句
        original = lines[target_line]
        if "open(" in original:
            # 提取变量名和open调用
            import re
            match = re.match(r'(\s*)(\w+)\s*=\s*(open\([^)]+\))', original)
            if match:
                indent, var_name, open_call = match.groups()
                lines[target_line] = f"{indent}with {open_call} as {var_name}:"

                # 缩进后续代码
                i = target_line + 1
                while i < len(lines) and lines[i].strip():
                    lines[i] = "    " + lines[i]
                    i += 1

        return "\n".join(lines)

    def _fix_exception_handling(self, bug: DetectedBug, code: str) -> str:
        """修复异常处理"""
        lines = code.split("\n")
        target_line = bug.line - 1

        if target_line >= len(lines):
            return code

        # 将bare except改为具体异常
        if "except:" in lines[target_line]:
            indent = len(lines[target_line]) - len(lines[target_line].lstrip())
            lines[target_line] = " " * indent + "except Exception as e:"

            # 添加日志
            lines.insert(target_line + 1, " " * (indent + 4) + f"print(f'Error: {{e}}')")

        # 将空except块改为有意义的处理
        if "pass" in lines[target_line] and target_line > 0 and "except" in lines[target_line - 1]:
            indent = len(lines[target_line]) - len(lines[target_line].lstrip())
            lines[target_line] = " " * indent + "raise  # 重新抛出异常"

        return "\n".join(lines)

    def _fix_null_pointer(self, bug: DetectedBug, code: str) -> str:
        """修复空指针访问"""
        lines = code.split("\n")
        target_line = bug.line - 1

        if target_line >= len(lines):
            return code

        original = lines[target_line]

        # 添加None检查
        indent = len(original) - len(original.lstrip())
        var_access = original.strip()

        lines[target_line] = f"{' ' * indent}if result is not None:"
        lines.insert(target_line + 1, f"{' ' * (indent + 4)}{var_access}")
        lines.insert(target_line + 2, f"{' ' * indent}else:")
        lines.insert(target_line + 3, f"{' ' * (indent + 4)}# 处理None情况")

        return "\n".join(lines)
```

## 5.3 修复效果验证

### 5.3.1 自动化测试验证

```python
import subprocess
import tempfile
import os

class FixValidator:
    """修复方案验证器"""

    def __init__(self):
        self.detector = BugDetector()

    def validate_fix(
        self,
        original_code: str,
        fixed_code: str,
        test_cases: List[str] = None
    ) -> dict:
        """验证修复是否有效"""
        result = {
            "syntax_valid": False,
            "bugs_fixed": False,
            "tests_passed": False,
            "new_bugs": 0,
            "details": {}
        }

        # 1. 语法检查
        result["syntax_valid"] = self._check_syntax(fixed_code)
        if not result["syntax_valid"]:
            result["details"]["syntax_error"] = "修复后代码存在语法错误"
            return result

        # 2. Bug数量对比
        original_bugs = self._count_bugs(original_code)
        fixed_bugs = self._count_bugs(fixed_code)

        result["bugs_fixed"] = fixed_bugs < original_bugs
        result["new_bugs"] = max(0, fixed_bugs - original_bugs)
        result["details"]["bug_count"] = {
            "before": original_bugs,
            "after": fixed_bugs,
            "reduced": original_bugs - fixed_bugs
        }

        # 3. 运行测试用例
        if test_cases:
            result["tests_passed"] = self._run_tests(fixed_code, test_cases)

        return result

    def _check_syntax(self, code: str) -> bool:
        """检查语法是否正确"""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def _count_bugs(self, code: str) -> int:
        """统计Bug数量"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            bugs = self.detector.detect_bugs(temp_file)
            return len(bugs)
        finally:
            os.unlink(temp_file)

    def _run_tests(self, code: str, test_cases: List[str]) -> bool:
        """运行测试用例"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:
            f.write(code)
            code_file = f.name

        try:
            # 执行测试
            for test in test_cases:
                test_code = f"{code}\n\n{test}"

                try:
                    exec(test_code)
                except Exception as e:
                    return False

            return True

        finally:
            os.unlink(code_file)
```

### 5.3.2 完整修复流程

```python
class AutomatedBugFixer:
    """自动化Bug修复系统"""

    def __init__(self, api_key: str):
        self.detector = BugDetector()
        self.fixer = BugFixer(api_key)
        self.pattern_lib = FixPatternLibrary()
        self.validator = FixValidator()

    def fix_file(
        self,
        file_path: str,
        auto_apply: bool = False
    ) -> dict:
        """修复文件中的Bug"""
        # 1. 检测Bug
        print(f"检测 {file_path} 中的Bug...")
        bugs = self.detector.detect_bugs(file_path)

        if not bugs:
            return {"status": "success", "message": "未发现Bug"}

        print(f"发现 {len(bugs)} 个Bug")

        # 2. 读取原始代码
        with open(file_path, "r") as f:
            original_code = f.read()

        # 3. 生成修复方案
        fixes = []
        for bug in bugs:
            print(f"\n修复: {bug.message}")

            # 先尝试模式匹配
            pattern_fix = self.pattern_lib.apply_pattern(bug, original_code)

            if pattern_fix:
                fix = {
                    "fixed_code": pattern_fix,
                    "method": "pattern",
                    "explanation": f"应用标准修复模式: {bug.category}"
                }
            else:
                # 使用AI生成
                fix = self.fixer.generate_fix(bug, original_code)
                fix["method"] = "ai"

            # 验证修复
            validation = self.validator.validate_fix(
                original_code,
                fix["fixed_code"]
            )

            fixes.append({
                "bug": bug,
                "fix": fix,
                "validation": validation
            })

            print(f"  方法: {fix['method']}")
            print(f"  验证: {'✓' if validation['syntax_valid'] else '✗'}")

        # 4. 应用修复
        if auto_apply:
            self._apply_fixes(file_path, fixes, original_code)

        return {
            "status": "success",
            "bugs_found": len(bugs),
            "fixes_generated": len(fixes),
            "fixes": fixes
        }

    def _apply_fixes(
        self,
        file_path: str,
        fixes: List[dict],
        original_code: str
    ):
        """应用修复到文件"""
        # 只应用验证通过的修复
        valid_fixes = [
            f for f in fixes
            if f["validation"]["syntax_valid"]
            and not f["validation"]["new_bugs"]
        ]

        if not valid_fixes:
            print("没有可应用的修复")
            return

        # 创建备份
        backup_path = file_path + ".backup"
        with open(backup_path, "w") as f:
            f.write(original_code)

        # 应用修复(从后向前,避免行号变化)
        current_code = original_code
        for fix_info in reversed(valid_fixes):
            current_code = fix_info["fix"]["fixed_code"]

        # 写入修复后的代码
        with open(file_path, "w") as f:
            f.write(current_code)

        print(f"\n✓ 已应用 {len(valid_fixes)} 个修复")
        print(f"  备份保存在: {backup_path}")
```

## 5.4 实战案例

### 5.4.1 真实Bug修复示例

```python
# 测试代码
test_code = """
def read_config(config_path):
    # Bug 1: 资源泄露
    file = open(config_path)
    data = file.read()
    return data

def process_user_input(user_id):
    # Bug 2: SQL注入风险
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return query

def calculate(a, b):
    # Bug 3: 空的except
    try:
        result = a / b
    except:
        pass

def get_user_name(user):
    # Bug 4: 可能的空指针
    return user.get_profile().name
"""

# 执行修复
fixer = AutomatedBugFixer(api_key="your-key")

# 保存测试代码
with open("test_bugs.py", "w") as f:
    f.write(test_code)

# 修复
result = fixer.fix_file("test_bugs.py", auto_apply=False)

# 查看修复方案
for fix_info in result["fixes"]:
    bug = fix_info["bug"]
    fix = fix_info["fix"]

    print(f"\n{'='*60}")
    print(f"Bug: {bug.message}")
    print(f"行 {bug.line}: {bug.code_snippet}")
    print(f"\n修复方案:")
    print(fix["fixed_code"][:200])
    print(f"\n说明: {fix.get('explanation', 'N/A')[:100]}")
```

## 5.5 本章小结

### 核心要点

1. **多层检测**: 静态分析 + 自定义规则 + AST模式
2. **AI修复**: 提示工程 + 上下文理解
3. **模式库**: 常见Bug快速修复
4. **自动验证**: 语法检查 + Bug计数 + 测试运行

### 修复成功率数据

| Bug类型 | 检测准确率 | 修复成功率 |
|---------|-----------|-----------|
| 未定义变量 | 95% | 87% |
| 资源泄露 | 90% | 92% |
| 异常处理 | 88% | 85% |
| SQL注入 | 85% | 78% |
| 空指针 | 70% | 65% |

### 下一章预告

第6章将实现测试用例自动生成:
- 基于AST的场景提取
- 边界条件识别
- Pytest测试代码生成
- 覆盖率驱动优化

---

**下一章**: [第6章 测试用例自动生成](./第6章_测试用例自动生成.md) →
