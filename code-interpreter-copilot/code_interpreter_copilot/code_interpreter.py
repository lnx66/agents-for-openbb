# worker.py
import sys
import io
from code import InteractiveConsole
from multiprocessing import Process, Queue
import pandas as pd


class CapturingInteractiveConsole(InteractiveConsole):
    def __init__(self, locals=None):
        super().__init__(locals=locals)
        self.output_buffer = io.StringIO()

    def runcode(self, code):
        old_stdout = sys.stdout
        old_displayhook = sys.displayhook
        sys.stdout = self.output_buffer

        def custom_displayhook(value):
            if value is not None:
                print(repr(value))

        sys.displayhook = custom_displayhook

        try:
            super().runcode(code)
        finally:
            sys.stdout = old_stdout
            sys.displayhook = old_displayhook

    def get_output(self):
        val = self.output_buffer.getvalue()
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)
        return val.rstrip("\n")

def repl_worker(input_queue: Queue, output_queue: Queue):
    console = CapturingInteractiveConsole()
    while True:
        code = input_queue.get()
        if code is None:
            break
        try:
            lines = code.strip().split('\n')
            if len(lines) > 1:
                # Execute all but the last line as statements
                stmt_code = '\n'.join(lines[:-1])
                if stmt_code:
                    compiled_stmt = compile(stmt_code, "<input>", "exec")
                    console.runcode(compiled_stmt)
                
                # Try to evaluate the last line for its result
                try:
                    compiled_expr = compile(lines[-1], "<input>", "eval")
                    result = eval(compiled_expr, console.locals)
                    if result is not None:
                        print(result, file=console.output_buffer)
                except SyntaxError:
                    compiled_stmt = compile(lines[-1], "<input>", "exec")
                    console.runcode(compiled_stmt)
            else:
                # Single line - try as statement first, then expression
                try:
                    compiled_stmt = compile(code, "<input>", "exec")
                    console.runcode(compiled_stmt)
                except SyntaxError:
                    # If it's not a valid statement, try as expression
                    compiled = compile(code, "<input>", "eval")
                    result = eval(compiled, console.locals)
                    if result is not None:
                        print(result, file=console.output_buffer)
            
            output = console.get_output()
            output_queue.put(output)
        except Exception as e:
            output_queue.put(str(e))