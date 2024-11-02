"""Console script for sisa_tree."""
import sisa_tree

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for sisa_tree."""
    console.print("Replace this message by putting your code into "
               "sisa_tree.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
