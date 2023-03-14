import typer

from fsd_path_planning.demo.json_demo import main

if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(main)
    app()
