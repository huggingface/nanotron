## Debugging the tests with vscode

To debug the tests with vscode, add the following json to your `launch.json` file.

```
{
    "name": "Test conversion",
    "type": "python",
        "request": "launch",
        "module": "pytest",
        "console": "integratedTerminal",
        "args": [
            "examples/llama/tests"
        ],
        "justMyCode": false
}
```
