from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone

from kino.models import KinoAnalysisState, KinoDraw


class Command(BaseCommand):
    help = "Live KINO update tick: sync latest draws and incrementally build analysis every N new games"

    def add_arguments(self, parser):
        parser.add_argument("--min-new-draws", type=int, default=10)
        parser.add_argument("--skip-sync", action="store_true")

    def handle(self, *args, **options):
        min_new_draws = options["min_new_draws"]
        skip_sync = options["skip_sync"]

        state, _ = KinoAnalysisState.objects.get_or_create(
            key="live_kino_tick",
            defaults={
                "value": {
                    "last_analyzed_draw_id": None,
                    "last_run_at": None,
                }
            },
        )

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Running live KINO tick..."))

        if not skip_sync:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Step 1: syncing latest draws..."))
            call_command("sync_kino_latest")

        latest_draw_id = (
            KinoDraw.objects.order_by("-draw_id")
            .values_list("draw_id", flat=True)
            .first()
        )

        if latest_draw_id is None:
            self.stdout.write(self.style.WARNING("No draws found."))
            return

        last_analyzed_draw_id = state.value.get("last_analyzed_draw_id")

        if last_analyzed_draw_id is None:
            new_draws = min_new_draws
        else:
            new_draws = latest_draw_id - last_analyzed_draw_id

        self.stdout.write("")
        self.stdout.write(f"Latest draw ID: {latest_draw_id}")
        self.stdout.write(f"Last analyzed draw ID: {last_analyzed_draw_id}")
        self.stdout.write(f"New draws: {new_draws}")

        if new_draws < min_new_draws:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Only {new_draws} new draws. Waiting for {min_new_draws}."
                )
            )
            return

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Step 2: building new window analysis..."))

        call_command("build_windows_incremental", window=20, step=10)
        call_command("build_windows_incremental", window=10, step=5)

        state.value = {
            "last_analyzed_draw_id": latest_draw_id,
            "last_run_at": timezone.now().isoformat(),
        }
        state.save()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Live KINO tick complete."))
        self.stdout.write(f"Analyzed through draw ID: {latest_draw_id}")