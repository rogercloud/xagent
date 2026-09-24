"""merge pending input and Google OAuth migrations

Revision ID: e24cebbf8cf5
Revises: 20260924_hide_google_drive_until_picker, 20260924_pending_injection
Create Date: 2026-09-24 15:39:03.037316

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "e24cebbf8cf5"
down_revision: Union[str, Sequence[str], None] = (
    "20260924_hide_google_drive_until_picker",
    "20260924_pending_injection",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
