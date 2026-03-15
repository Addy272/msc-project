"""
Command-line helper for managing admin users.

Usage:
    python manage_admin.py --username admin
    python manage_admin.py --username admin --password "new-password"
"""

import argparse
import getpass

from flask import Flask

from config import Config
from database.models import AdminUser, db


def _prompt_for_password():
    """Prompt for a password twice to avoid accidental typos."""
    password = getpass.getpass('Password: ')
    confirm_password = getpass.getpass('Confirm password: ')

    if not password:
        raise ValueError('Password cannot be empty.')
    if password != confirm_password:
        raise ValueError('Passwords do not match.')

    return password


def create_app():
    """Create a lightweight Flask app for database access."""
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    return app


def upsert_admin_user(username, password, active=True):
    """Create a new admin user or update an existing one."""
    admin_user = AdminUser.find_by_username(username)

    if admin_user is None:
        admin_user = AdminUser(username=username.strip(), is_active=active)
        db.session.add(admin_user)
        action = 'created'
    else:
        admin_user.is_active = active
        action = 'updated'

    admin_user.set_password(password)
    db.session.commit()
    return action, admin_user


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Create or update an admin user.')
    parser.add_argument('--username', required=True, help='Admin username to create or update')
    parser.add_argument('--password', help='Admin password. If omitted, you will be prompted securely.')
    parser.add_argument(
        '--inactive',
        action='store_true',
        help='Store the account as inactive instead of active.',
    )
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()
    username = args.username.strip()

    if not username:
        raise ValueError('Username cannot be empty.')

    password = args.password or _prompt_for_password()
    app = create_app()

    with app.app_context():
        db.create_all()
        action, admin_user = upsert_admin_user(username, password, active=not args.inactive)
        status = 'inactive' if args.inactive else 'active'
        print(f"Admin user '{admin_user.username}' {action} successfully ({status}).")


if __name__ == '__main__':
    main()
