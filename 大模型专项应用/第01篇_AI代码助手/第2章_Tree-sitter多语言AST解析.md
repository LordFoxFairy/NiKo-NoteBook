# 第2章 Tree-sitter多语言AST解析

> 从语法树到代码理解:掌握现代解析器的核心技术

## 2.1 Tree-sitter核心原理

### 2.1.1 什么是Tree-sitter?

**定义**: Tree-sitter是一个解析器生成器(Parser Generator)和增量解析库,专为代码编辑器设计。

**核心特性**:
```
1. 增量解析(Incremental Parsing)
   - 仅重新解析文件变更部分
   - 性能提升10-100倍

2. 错误恢复(Error Recovery)
   - 语法错误时仍能生成部分AST
   - 不会因局部错误导致整体解析失败

3. 多语言统一接口
   - 一套API支持50+语言
   - 通过Language对象切换语言

4. 零依赖运行时
   - 纯C11实现
   - 生成的解析器无外部依赖
```

### 2.1.2 AST vs CST

**对比传统AST**:

| 特性 | 传统AST (如Python ast模块) | Tree-sitter CST |
|------|---------------------------|-----------------|
| 结构 | 抽象语法树(忽略细节) | 具体语法树(保留所有token) |
| 空格/注释 | 丢弃 | 保留 |
| 括号/分号 | 丢弃 | 保留 |
| 原始代码重建 | 不可能 | 完全可能 |
| 适用场景 | 编译/执行 | 编辑器/代码分析 |

**示例对比**:

```python
# 原始代码
result = (a + b) * 2  # 计算结果

# 传统AST (简化表示)
Assign(
    targets=[Name(id='result')],
    value=BinOp(
        left=BinOp(left=Name('a'), op=Add(), right=Name('b')),
        op=Mult(),
        right=Constant(2)
    )
)
# 注释和括号已丢失!

# Tree-sitter CST (简化表示)
assignment(
    left: identifier('result'),
    operator: '=',
    right: binary_expression(
        left: parenthesized_expression(
            '(',
            binary_expression(
                left: identifier('a'),
                operator: '+',
                right: identifier('b')
            ),
            ')'
        ),
        operator: '*',
        right: integer(2)
    ),
    comment: '# 计算结果'
)
# 所有细节都保留!
```

**为什么代码助手需要CST?**
1. 代码格式化需要保留空格/注释
2. 重构时需要精确修改特定位置
3. 语法高亮需要知道每个token的类型

### 2.1.3 增量解析机制

**原理**:

```
初始状态: 解析整个文件
┌──────────────────────────────────┐
│ def foo():              │ [已解析]
│     if bar:             │ [已解析]
│         return 1        │ [已解析]
└──────────────────────────────────┘

修改: 用户在第2行末尾添加 "and baz"
┌──────────────────────────────────┐
│ def foo():              │ [未变更 → 复用]
│     if bar and baz:     │ [已变更 → 重新解析]
│         return 1        │ [未变更 → 复用]
└──────────────────────────────────┘

性能提升: 仅解析1行,而非3行
```

**技术实现**:

```python
from tree_sitter import Parser, Language, Tree
import tree_sitter_python as tspython

class IncrementalParser:
    def __init__(self):
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        self.old_tree = None  # 保存旧的语法树

    def parse_initial(self, code: bytes) -> Tree:
        """首次解析"""
        tree = self.parser.parse(code)
        self.old_tree = tree
        return tree

    def parse_incremental(
        self,
        new_code: bytes,
        edit_start_byte: int,
        edit_old_end_byte: int,
        edit_new_end_byte: int,
        edit_start_point: tuple,
        edit_old_end_point: tuple,
        edit_new_end_point: tuple
    ) -> Tree:
        """增量解析"""
        if self.old_tree is None:
            return self.parse_initial(new_code)

        # 1. 告诉旧树发生了什么变更
        self.old_tree.edit(
            start_byte=edit_start_byte,
            old_end_byte=edit_old_end_byte,
            new_end_byte=edit_new_end_byte,
            start_point=edit_start_point,
            old_end_point=edit_old_end_point,
            new_end_point=edit_new_end_point,
        )

        # 2. 基于旧树进行增量解析
        new_tree = self.parser.parse(new_code, self.old_tree)

        # 3. 更新旧树引用
        self.old_tree = new_tree

        return new_tree
```

**实际应用示例**:

```python
# 场景: 用户编辑代码
original_code = b"def foo():\n    return 42"
modified_code = b"def foo():\n    return 42 + 1"

parser = IncrementalParser()

# 首次解析
tree1 = parser.parse_initial(original_code)
print(f"首次解析耗时: {tree1.root_node}")

# 用户在末尾添加 " + 1"
# 修改位置: 第1行,第15字节(return 42之后)
tree2 = parser.parse_incremental(
    new_code=modified_code,
    edit_start_byte=21,        # "2"之后的位置
    edit_old_end_byte=21,      # 旧代码结束位置
    edit_new_end_byte=25,      # 新代码结束位置
    edit_start_point=(1, 14),  # (行, 列)
    edit_old_end_point=(1, 14),
    edit_new_end_point=(1, 18)
)

# Tree-sitter自动识别仅需重新解析return语句
# 其他节点(def, foo, ())全部复用
```

**性能数据**:

| 文件大小 | 全量解析 | 增量解析 | 提升 |
|---------|---------|---------|-----|
| 100行 | 2ms | 0.3ms | 6.7x |
| 1000行 | 15ms | 0.5ms | 30x |
| 10000行 | 180ms | 1.2ms | 150x |

## 2.2 Python代码解析实战

### 2.2.1 基础解析

**安装依赖**:
```bash
pip install tree-sitter==0.25.2 tree-sitter-python==0.23.12
```

**最小示例**:

```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# 1. 加载语言
PY_LANGUAGE = Language(tspython.language())

# 2. 创建解析器
parser = Parser(PY_LANGUAGE)

# 3. 解析代码
code = b"""
def calculate_sum(a, b):
    \"\"\"计算两数之和\"\"\"
    result = a + b
    return result
"""

tree = parser.parse(code)

# 4. 访问根节点
root = tree.root_node
print(root.type)  # 输出: module
print(root.sexp())  # 输出: S-expression格式的完整语法树
```

**输出的S-expression**:
```scheme
(module
  (function_definition
    name: (identifier)
    parameters: (parameters (identifier) (identifier))
    body: (block
      (expression_statement (string))  ; 文档字符串
      (assignment
        left: (identifier)
        right: (binary_operator
          left: (identifier)
          operator: +
          right: (identifier)))
      (return_statement (identifier)))))
```

### 2.2.2 遍历AST

**方法1: 递归遍历**

```python
def traverse_tree(node, depth=0):
    """递归遍历所有节点"""
    indent = "  " * depth
    print(f"{indent}{node.type} [{node.start_point}-{node.end_point}]")

    for child in node.children:
        traverse_tree(child, depth + 1)

# 使用
traverse_tree(tree.root_node)
```

**输出示例**:
```
module [(0, 0)-(5, 0)]
  function_definition [(1, 0)-(5, 0)]
    def [(1, 0)-(1, 3)]
    identifier [(1, 4)-(1, 17)]
    parameters [(1, 17)-(1, 23)]
      ( [(1, 17)-(1, 18)]
      identifier [(1, 18)-(1, 19)]
      , [(1, 19)-(1, 20)]
      identifier [(1, 21)-(1, 22)]
      ) [(1, 22)-(1, 23)]
    : [(1, 23)-(1, 24)]
    block [(2, 4)-(5, 0)]
      ...
```

**方法2: TreeCursor高效遍历**

```python
from tree_sitter import TreeCursor

def traverse_with_cursor(tree):
    """使用TreeCursor遍历(性能更好)"""
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            print(f"{cursor.node.type} at {cursor.node.start_point}")

            # 尝试进入第一个子节点
            if cursor.goto_first_child():
                continue

        # 尝试移动到下一个兄弟节点
        if cursor.goto_next_sibling():
            visited_children = False
            continue

        # 回到父节点
        if not cursor.goto_parent():
            break

        visited_children = True
```

**性能对比**:
- 递归遍历: 简单直观,内存占用高(大文件可能栈溢出)
- TreeCursor: 性能更好,内存占用低(适合大文件)

### 2.2.3 提取特定信息

**场景1: 提取所有函数定义**

```python
def extract_functions(tree):
    """提取所有函数名、参数、返回类型"""
    functions = []

    def visit_node(node):
        if node.type == "function_definition":
            # 提取函数名
            name_node = node.child_by_field_name("name")
            func_name = code[name_node.start_byte:name_node.end_byte].decode()

            # 提取参数
            params_node = node.child_by_field_name("parameters")
            params = []
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        param_name = code[child.start_byte:child.end_byte].decode()
                        params.append(param_name)

            # 提取返回类型(如果有类型注解)
            return_type = None
            for child in node.children:
                if child.type == "type":
                    return_type = code[child.start_byte:child.end_byte].decode()

            functions.append({
                "name": func_name,
                "params": params,
                "return_type": return_type,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0]
            })

        # 递归访问子节点
        for child in node.children:
            visit_node(child)

    visit_node(tree.root_node)
    return functions

# 测试
code = b"""
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str):
    print(f"Hello, {name}")

class Calculator:
    def multiply(self, x, y):
        return x * y
"""

tree = parser.parse(code)
functions = extract_functions(tree)

for func in functions:
    print(f"函数: {func['name']}")
    print(f"  参数: {func['params']}")
    print(f"  返回类型: {func['return_type']}")
    print(f"  位置: {func['start_line']}-{func['end_line']}")
    print()
```

**输出**:
```
函数: add
  参数: ['a', 'b']
  返回类型: int
  位置: 1-2

函数: greet
  参数: ['name']
  返回类型: None
  位置: 4-5

函数: multiply
  参数: ['self', 'x', 'y']
  返回类型: None
  位置: 8-9
```

**场景2: 提取所有导入语句**

```python
def extract_imports(tree, code):
    """提取import和from...import语句"""
    imports = []

    def visit_node(node):
        if node.type == "import_statement":
            # import os, sys
            for child in node.children:
                if child.type == "dotted_name":
                    module = code[child.start_byte:child.end_byte].decode()
                    imports.append({"type": "import", "module": module})

        elif node.type == "import_from_statement":
            # from pathlib import Path
            module = None
            names = []

            for child in node.children:
                if child.type == "dotted_name":
                    module = code[child.start_byte:child.end_byte].decode()
                elif child.type == "identifier":
                    name = code[child.start_byte:child.end_byte].decode()
                    names.append(name)

            imports.append({
                "type": "from_import",
                "module": module,
                "names": names
            })

        for child in node.children:
            visit_node(child)

    visit_node(tree.root_node)
    return imports

# 测试
code = b"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
"""

tree = parser.parse(code)
imports = extract_imports(tree, code)

for imp in imports:
    if imp["type"] == "import":
        print(f"import {imp['module']}")
    else:
        print(f"from {imp['module']} import {', '.join(imp['names'])}")
```

**场景3: 检测潜在Bug**

```python
def detect_division_by_zero(tree, code):
    """检测可能的除零错误"""
    warnings = []

    def visit_node(node):
        if node.type == "binary_operator":
            # 检查是否是除法运算
            operator_node = None
            right_node = None

            for child in node.children:
                if child.type == "/":
                    operator_node = child
                elif operator_node:  # 运算符之后的是右操作数
                    right_node = child
                    break

            if operator_node and right_node:
                # 检查右操作数是否是字面量0
                if right_node.type == "integer" or right_node.type == "float":
                    value = code[right_node.start_byte:right_node.end_byte].decode()
                    if value == "0" or value == "0.0":
                        warnings.append({
                            "message": "除零错误",
                            "line": node.start_point[0] + 1,
                            "code": code[node.start_byte:node.end_byte].decode()
                        })

        for child in node.children:
            visit_node(child)

    visit_node(tree.root_node)
    return warnings

# 测试
code = b"""
result1 = 10 / 2    # 正常
result2 = 10 / 0    # Bug!
result3 = x / y     # 无法静态判断
"""

tree = parser.parse(code)
warnings = detect_division_by_zero(tree, code)

for warning in warnings:
    print(f"行 {warning['line']}: {warning['message']}")
    print(f"  代码: {warning['code']}")
```

## 2.3 Query语法高级用法

### 2.3.1 Query语法基础

**什么是Query?**
Tree-sitter Query是一种声明式模式匹配语言,类似于XPath但专为语法树设计。

**基础语法**:

```scheme
; 匹配所有函数定义
(function_definition) @function

; 匹配特定名称的函数
(function_definition
  name: (identifier) @func_name
  (#eq? @func_name "main"))

; 匹配带有文档字符串的函数
(function_definition
  body: (block
    (expression_statement (string) @docstring) .))
```

**Python API使用**:

```python
from tree_sitter import Query

# 1. 编译Query
query_source = """
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)
"""

query = PY_LANGUAGE.query(query_source)

# 2. 执行Query
code = b"""
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""

tree = parser.parse(code)
captures = query.captures(tree.root_node)

# 3. 处理结果
for node, capture_name in captures:
    print(f"{capture_name}: {code[node.start_byte:node.end_byte].decode()}")
```

**输出**:
```
function.name: add
function.params: (a, b)
function.body: return a + b
function.name: multiply
function.params: (x, y)
function.body: return x * y
```

### 2.3.2 高级Query模式

**模式1: 匹配嵌套结构**

```scheme
; 找出所有在if语句内部的return语句
(if_statement
  condition: (_)
  consequence: (block
    (return_statement) @early_return))
```

**模式2: 使用谓词过滤**

```scheme
; 找出所有以"test_"开头的函数
(function_definition
  name: (identifier) @test_func
  (#match? @test_func "^test_"))

; 找出所有未使用的变量(赋值但从未读取)
(assignment
  left: (identifier) @unused_var
  (#not-match? @unused_var "^_"))  ; 排除下划线开头
```

**模式3: 捕获父子关系**

```scheme
; 找出所有调用特定函数的地方
(call
  function: (identifier) @func_name
  (#eq? @func_name "dangerous_function")
  arguments: (argument_list) @args)
```

**实战案例: 查找安全问题**

```python
def find_sql_injection_risks(tree, code):
    """查找可能的SQL注入风险"""
    # Query: 查找字符串拼接构建的SQL语句
    query_source = """
    (call
      function: (attribute
        object: (identifier) @db_obj
        attribute: (identifier) @method)
      arguments: (argument_list
        (binary_operator
          left: (string) @sql_string
          operator: "+"
          right: (_)) @concat))
    """

    query = PY_LANGUAGE.query(query_source)
    captures = query.captures(tree.root_node)

    risks = []
    for node, name in captures:
        if name == "sql_string":
            sql_text = code[node.start_byte:node.end_byte].decode()
            if any(keyword in sql_text.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                risks.append({
                    "line": node.start_point[0] + 1,
                    "message": "可能的SQL注入风险: 使用字符串拼接构建SQL",
                    "code": code[node.parent.parent.start_byte:node.parent.parent.end_byte].decode()
                })

    return risks

# 测试
code = b"""
# 危险做法
user_id = request.GET['id']
query = "SELECT * FROM users WHERE id = " + user_id  # 风险!
cursor.execute(query)

# 安全做法
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))
"""

tree = parser.parse(code)
risks = find_sql_injection_risks(tree, code)

for risk in risks:
    print(f"行 {risk['line']}: {risk['message']}")
```

### 2.3.3 Query性能优化

**技巧1: 使用锚点(Anchors)**

```scheme
; 慢: 遍历整个树
(function_definition) @func

; 快: 只在模块顶层查找
(module
  (function_definition) @func)
```

**技巧2: 精确匹配vs模糊匹配**

```scheme
; 慢: 匹配所有调用
(call) @all_calls

; 快: 只匹配特定函数
(call
  function: (identifier) @func_name
  (#eq? @func_name "critical_function"))
```

**技巧3: 缓存Query对象**

```python
class QueryCache:
    def __init__(self, language):
        self.language = language
        self._cache = {}

    def get_query(self, query_source: str) -> Query:
        """获取或创建Query对象"""
        if query_source not in self._cache:
            self._cache[query_source] = self.language.query(query_source)
        return self._cache[query_source]

# 使用
cache = QueryCache(PY_LANGUAGE)
query = cache.get_query("(function_definition) @func")  # 首次编译
query = cache.get_query("(function_definition) @func")  # 从缓存获取
```

## 2.4 JavaScript代码解析

### 2.4.1 安装与配置

```bash
pip install tree-sitter-javascript==0.23.14
```

```python
import tree_sitter_javascript as tsjavascript

JS_LANGUAGE = Language(tsjavascript.language())
js_parser = Parser(JS_LANGUAGE)
```

### 2.4.2 JavaScript特有节点类型

```python
code = b"""
// 箭头函数
const add = (a, b) => a + b;

// 异步函数
async function fetchData() {
    const response = await fetch('/api/data');
    return response.json();
}

// 模板字符串
const greeting = `Hello, ${name}!`;

// 解构赋值
const { x, y } = point;
const [first, ...rest] = array;

// JSX (需要tree-sitter-javascript的jsx支持)
const element = <div className="container">Content</div>;
"""

tree = js_parser.parse(code)

# 查找箭头函数
query = JS_LANGUAGE.query("""
(arrow_function
  parameters: (formal_parameters) @params
  body: (_) @body) @arrow_func
""")

captures = query.captures(tree.root_node)
for node, name in captures:
    if name == "arrow_func":
        print(f"箭头函数: {code[node.start_byte:node.end_byte].decode()}")
```

### 2.4.3 实战: 检测异步反模式

```python
def find_unhandled_promises(tree, code):
    """查找未处理的Promise(可能导致静默失败)"""
    query = JS_LANGUAGE.query("""
    (call_expression
      function: (member_expression
        object: (_)
        property: (property_identifier) @method)
      (#match? @method "^(then|catch)$"))
    """)

    # 找到所有.then()和.catch()调用
    promise_calls = set()
    captures = query.captures(tree.root_node)
    for node, _ in captures:
        promise_calls.add(node.parent.start_byte)

    # 查找所有Promise相关的调用
    all_promise_query = JS_LANGUAGE.query("""
    (call_expression
      function: (identifier) @func
      (#match? @func "^(fetch|axios|Promise)$"))
    """)

    issues = []
    for node, _ in all_promise_query.captures(tree.root_node):
        call_node = node.parent
        # 检查是否有.then()或.catch()处理
        if call_node.start_byte not in promise_calls:
            # 检查是否在await表达式中
            parent = call_node.parent
            if parent and parent.type != "await_expression":
                issues.append({
                    "line": call_node.start_point[0] + 1,
                    "message": "Promise未处理: 可能导致错误被忽略",
                    "code": code[call_node.start_byte:call_node.end_byte].decode()
                })

    return issues
```

## 2.5 Go代码解析

### 2.5.1 安装与配置

```bash
pip install tree-sitter-go==0.23.5
```

```python
import tree_sitter_go as tsgo

GO_LANGUAGE = Language(tsgo.language())
go_parser = Parser(GO_LANGUAGE)
```

### 2.5.2 Go特有场景

```python
code = b"""
package main

import (
    "fmt"
    "errors"
)

// 接口定义
type Reader interface {
    Read(p []byte) (n int, err error)
}

// 结构体与方法
type File struct {
    name string
}

func (f *File) Read(p []byte) (int, error) {
    return 0, errors.New("not implemented")
}

// Goroutine
func processData() {
    go func() {
        fmt.Println("Running in goroutine")
    }()
}

// Defer语句
func openFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()

    // 处理文件...
    return nil
}
"""

tree = go_parser.parse(code)
```

### 2.5.3 实战: 检测Goroutine泄露

```python
def find_goroutine_leaks(tree, code):
    """查找可能导致泄露的goroutine"""
    query = GO_LANGUAGE.query("""
    (go_statement
      (call_expression
        function: (func_literal) @goroutine_func))
    """)

    leaks = []
    for node, _ in query.captures(tree.root_node):
        # 检查goroutine内是否有死循环或阻塞调用
        func_body = node.child_by_field_name("body")
        if func_body:
            # 简单检查: 是否有for{}或select{}
            body_text = code[func_body.start_byte:func_body.end_byte].decode()
            if "for {" in body_text and "return" not in body_text:
                leaks.append({
                    "line": node.start_point[0] + 1,
                    "message": "可能的Goroutine泄露: 无限循环且无退出条件",
                    "code": code[node.start_byte:node.end_byte].decode()[:100]
                })

    return leaks
```

## 2.6 跨语言代码导航

### 2.6.1 统一的符号提取接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class Symbol:
    name: str
    kind: str  # function/class/variable/interface
    start_line: int
    end_line: int
    children: List['Symbol'] = None

class LanguageAnalyzer(ABC):
    @abstractmethod
    def extract_symbols(self, tree, code: bytes) -> List[Symbol]:
        """提取代码符号"""
        pass

class PythonAnalyzer(LanguageAnalyzer):
    def extract_symbols(self, tree, code: bytes) -> List[Symbol]:
        symbols = []

        def visit(node):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                symbols.append(Symbol(
                    name=code[name_node.start_byte:name_node.end_byte].decode(),
                    kind="function",
                    start_line=node.start_point[0],
                    end_line=node.end_point[0]
                ))
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                symbols.append(Symbol(
                    name=code[name_node.start_byte:name_node.end_byte].decode(),
                    kind="class",
                    start_line=node.start_point[0],
                    end_line=node.end_point[0]
                ))

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return symbols

class JavaScriptAnalyzer(LanguageAnalyzer):
    def extract_symbols(self, tree, code: bytes) -> List[Symbol]:
        # 实现JavaScript符号提取逻辑
        pass

class GoAnalyzer(LanguageAnalyzer):
    def extract_symbols(self, tree, code: bytes) -> List[Symbol]:
        # 实现Go符号提取逻辑
        pass
```

### 2.6.2 多语言项目分析

```python
class MultiLanguageAnalyzer:
    def __init__(self):
        self.analyzers = {
            ".py": (Parser(PY_LANGUAGE), PythonAnalyzer()),
            ".js": (Parser(JS_LANGUAGE), JavaScriptAnalyzer()),
            ".go": (Parser(GO_LANGUAGE), GoAnalyzer()),
        }

    def analyze_project(self, project_path: str) -> dict:
        """分析整个项目"""
        from pathlib import Path

        all_symbols = {}

        for ext, (parser, analyzer) in self.analyzers.items():
            files = list(Path(project_path).rglob(f"*{ext}"))

            for file_path in files:
                with open(file_path, "rb") as f:
                    code = f.read()

                tree = parser.parse(code)
                symbols = analyzer.extract_symbols(tree, code)

                all_symbols[str(file_path)] = symbols

        return all_symbols

    def find_symbol(self, project_path: str, symbol_name: str) -> List[dict]:
        """全项目符号搜索"""
        all_symbols = self.analyze_project(project_path)

        results = []
        for file_path, symbols in all_symbols.items():
            for symbol in symbols:
                if symbol_name in symbol.name:
                    results.append({
                        "file": file_path,
                        "symbol": symbol
                    })

        return results
```

## 2.7 性能优化与最佳实践

### 2.7.1 缓存策略

```python
import hashlib
from typing import Optional

class CachedParser:
    def __init__(self, language):
        self.parser = Parser(language)
        self._cache = {}  # 哈希 -> (Tree, 代码)

    def parse(self, code: bytes) -> Tree:
        """带缓存的解析"""
        code_hash = hashlib.md5(code).hexdigest()

        if code_hash in self._cache:
            return self._cache[code_hash][0]

        tree = self.parser.parse(code)
        self._cache[code_hash] = (tree, code)

        # 限制缓存大小
        if len(self._cache) > 100:
            # 删除最旧的项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return tree
```

### 2.7.2 并行处理

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def parse_file(file_path: str) -> dict:
    """解析单个文件(在独立进程中)"""
    with open(file_path, "rb") as f:
        code = f.read()

    # 这里需要重新创建parser(不能跨进程共享)
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(code)

    return {
        "file": file_path,
        "node_count": len(list(tree.root_node.children))
    }

def parse_project_parallel(project_path: str, max_workers: int = 4):
    """并行解析整个项目"""
    files = list(Path(project_path).rglob("*.py"))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(parse_file, [str(f) for f in files])

    return list(results)
```

### 2.7.3 内存优化

```python
class StreamingParser:
    """流式处理大文件"""

    def __init__(self, parser):
        self.parser = parser

    def parse_large_file(self, file_path: str, chunk_size: int = 1024 * 1024):
        """分块处理大文件"""
        with open(file_path, "rb") as f:
            # 读取文件头部
            chunk = f.read(chunk_size)

            # 先解析可用部分
            tree = self.parser.parse(chunk)

            # 继续读取剩余部分(如果需要)
            while True:
                next_chunk = f.read(chunk_size)
                if not next_chunk:
                    break

                # 使用增量解析更新树
                chunk += next_chunk
                tree = self.parser.parse(chunk, tree)

            return tree
```

## 2.8 本章小结

### 核心要点

1. **Tree-sitter是CST而非AST**
   - 保留所有语法细节(空格/注释/括号)
   - 适合代码编辑器场景

2. **增量解析是性能关键**
   - 仅重新解析变更部分
   - 10-150x性能提升

3. **Query语法强大**
   - 声明式模式匹配
   - 支持复杂的结构查询

4. **多语言统一接口**
   - 一套API支持50+语言
   - 易于构建跨语言工具

5. **性能优化策略**
   - 缓存解析结果
   - 并行处理多文件
   - Query对象复用

### 实战练习

1. **基础**: 编写一个工具提取Python文件中所有TODO注释
2. **中级**: 实现一个跨语言的"查找所有函数调用"功能
3. **高级**: 构建一个代码复杂度分析器(圈复杂度计算)

### 下一章预告

第3章将深入静态分析工具集成:
- Ruff的10-100x性能提升如何实现
- mypy类型检查与AST的结合
- Bandit安全扫描的原理
- 多工具结果聚合策略

---

**参考资源**:
- Tree-sitter官方文档: https://tree-sitter.github.io/tree-sitter/
- py-tree-sitter API: https://tree-sitter.github.io/py-tree-sitter/
- Query语法参考: https://tree-sitter.github.io/tree-sitter/using-parsers#query-syntax
- 语言解析器列表: https://github.com/tree-sitter

**下一章**: [第3章 静态分析工具集成](./第3章_静态分析工具集成.md) →
