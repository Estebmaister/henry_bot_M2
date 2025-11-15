#!/usr/bin/env python3
"""
Henry Bot M2 - Main Entry Point

This file serves as the module orchestrator only.
All business logic is delegated to specialized modules.
"""

from src.modules.logging.logger import setup_logging
from src.core.exceptions import HenryBotError
from src.core.agent import HenryBot
from src.core.config import settings
import sys
import json
import uvicorn
import logging
import asyncio
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "logs",
        "data/documents",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point - orchestrates module connections."""
    try:
        # Check command line arguments first
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
        else:
            command = None

        # Setup logging (console only for server mode)
        if command == "server":
            setup_logging()
        else:
            # For CLI, only setup file logging (silent console)
            log_dir = Path(settings.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.WARNING,  # Only warnings/errors to console
                format='%(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(settings.log_file),
                ]
            )

        # Create necessary directories
        setup_directories()

        if command == "server":
            # Start the API server
            print("🚀 Starting Henry Bot M2 API Server...")
            print(
                f"📍 Server will be available at: http://{settings.host}:{settings.port}")
            print(
                f"📚 API Documentation: http://{settings.host}:{settings.port}/docs")

            # Lazy import of API server components only when needed
            from src.modules.api.server import create_app
            app = create_app()
            uvicorn.run(
                app,
                host=settings.host,
                port=settings.port,
                log_level=settings.log_level.lower()
            )

        elif command == "cli":
            # CLI interface (inherited from M1)
            if len(sys.argv) < 3:
                print("""
🧰 Henry Bot M2 - CLI Interface

Usage:
  python -m src.main cli "Your question here"

Example:
  python -m src.main cli "What is the capital of Spain?"

API Server:
  python -m src.main server
""")
                sys.exit(1)

            # Process the question
            user_question = " ".join(sys.argv[2:])
            bot = HenryBot()
            result = asyncio.run(bot.process_question(user_question))
            print(json.dumps(result, indent=2))

        elif command == "status":
            # Show system status
            bot = HenryBot()
            status = bot.get_system_status()
            print(json.dumps(status, indent=2))

        else:
            # Show usage information
            print("""
🧰 Henry Bot M2 - Enhanced LLM Agent with RAG

Commands:
  server          Start the API server
  cli <question>  Process a single question via CLI
  status          Show system status

Examples:
  python -m src.main server
  python -m src.main cli "What is the capital of Spain?"
  python -m src.main status
""")
            sys.exit(1)

    except HenryBotError as e:
        print(f"❌ Henry Bot Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
