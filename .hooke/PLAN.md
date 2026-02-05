# Plan for Task 633de9ba

**Generated:** 2026-02-05T18:34:13.880076+00:00  
**Task ID:** 633de9bae39e426eaa2674d31c062fc9  
**Run ID:** task-20260205-633de9ba  
**Model:** moonshotai/kimi-k2.5

---

## Task Prompt
在仓库里新增 scripts/evaluate_linreg.py：读取一个 metrics.jsonl，打印 min_loss 和 last_loss。不要引入任何第三方依赖。更新 README 加用法。最后运行 python3 scripts/run_tests.py 验证。

## Assumptions
- `metrics.jsonl` contains JSON lines with numeric `loss` field
- Repository has existing `scripts/` directory structure
- `scripts/run_tests.py` exists and validates new additions
- Python 3.7+ available in environment
- README is Markdown format and documents CLI tools

## First Checks
1. Verify `scripts/` directory exists in worktree root
2. Inspect sample `metrics.jsonl` to confirm field names (loss vs train_loss, etc.)
3. Check `scripts/run_tests.py` to understand validation criteria (exit codes, output format)
4. Review existing script headers/code style for consistency
5. Confirm README section structure for adding usage docs

## Proposed Steps
1. **Create evaluation script**: Write `scripts/evaluate_linreg.py` using only `json` and `sys` stdlib modules
   - Parse JSONL from stdin or file argument
   - Track minimum loss and final loss across all lines
   - Print `min_loss: {value}` and `last_loss: {value}` to stdout
   - Exit with code 0 on success, 1 on file/parse errors
2. **Update documentation**: Append usage to README.md under Scripts or Usage section
   - Add command example: `python3 scripts/evaluate_linreg.py path/to/metrics.jsonl`
   - Document expected output format
3. **Validate implementation**: Execute `python3 scripts/run_tests.py`
   - Fix any linting or functional test failures
   - Ensure script handles empty files gracefully

## Risks & Approvals
- **Schema ambiguity**: Confirm exact field name in metrics.jsonl (loss/val_loss/train_loss) before implementing
- **Test dependency**: `run_tests.py` may expect specific argparse interface or output formatting; review first
- **File path**: Determine if script accepts file path as argument or reads from fixed location
- **Approval required**: None for standard library usage; confirm stdout format matches project conventions
