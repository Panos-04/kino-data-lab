from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone

from kino.models import KinoDraw, KinoAnalysisState


class Command(BaseCommand):
    help = "Refresh KINO data and rebuild analysis when enough new draws exist"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=90)
        parser.add_argument("--min-new-draws", type=int, default=10)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--skip-heavy", action="store_true")

    def handle(self, *args, **options):
        days = options["days"]
        min_new_draws = options["min_new_draws"]
        force = options["force"]
        skip_heavy = options["skip_heavy"]

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Refreshing KINO Lab..."))
        self.stdout.write(f"Keeping recent days: {days}")
        self.stdout.write(f"Minimum new draws before rebuild: {min_new_draws}")
        self.stdout.write(f"Force rebuild: {force}")
        self.stdout.write(f"Skip heavy analysis: {skip_heavy}")

        state, _ = KinoAnalysisState.objects.get_or_create(
            key="main_refresh",
            defaults={
                "value": {
                    "last_analyzed_draw_id": None,
                    "last_refresh_time": None,
                }
            },
        )

        previous_latest_draw_id = (
            KinoDraw.objects.order_by("-draw_id").values_list("draw_id", flat=True).first()
        )

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Step 1: syncing recent KINO draws..."))

        call_command("sync_kino_recent", days=days)

        latest_draw_id = (
            KinoDraw.objects.order_by("-draw_id").values_list("draw_id", flat=True).first()
        )

        if latest_draw_id is None:
            self.stdout.write(self.style.WARNING("No draws found after sync."))
            return

        last_analyzed_draw_id = state.value.get("last_analyzed_draw_id")

        if last_analyzed_draw_id is None:
            new_draws_since_analysis = None
        else:
            new_draws_since_analysis = latest_draw_id - last_analyzed_draw_id

        self.stdout.write("")
        self.stdout.write(f"Previous latest draw ID: {previous_latest_draw_id}")
        self.stdout.write(f"Current latest draw ID: {latest_draw_id}")
        self.stdout.write(f"Last analyzed draw ID: {last_analyzed_draw_id}")
        self.stdout.write(f"New draws since analysis: {new_draws_since_analysis}")

        should_rebuild = (
            force
            or last_analyzed_draw_id is None
            or new_draws_since_analysis is None
            or new_draws_since_analysis >= min_new_draws
        )

        if not should_rebuild:
            self.stdout.write("")
            self.stdout.write(
                self.style.SUCCESS(
                    f"Only {new_draws_since_analysis} new draws. Skipping rebuild."
                )
            )
            return

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Step 2: rebuilding window analysis..."))

        call_command("build_windows", window=20, step=10, rebuild=True)
        call_command("build_windows", window=10, step=5, rebuild=True)

        if not skip_heavy:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Step 3: rebuilding shape events..."))

            call_command("build_shape_events", shape="all", rebuild=True)

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Step 4: rebuilding shape movements..."))

            movement_shapes = [
                "cross",
                "box_2x2",
                "vertical_4",
                "horizontal_4",
                "diagonal_down_4",
                "diagonal_up_4",
                "l_shape",
            ]

            for shape in movement_shapes:
                self.stdout.write(f"Building movements for {shape}...")

                call_command(
                    "build_shape_movements",
                    shape=shape,
                    min_hits=4,
                    future=10,
                    mode="one-to-one",
                    rebuild=True,
                )

        state.value = {
            "last_analyzed_draw_id": latest_draw_id,
            "last_refresh_time": timezone.now().isoformat(),
            "days": days,
            "skip_heavy": skip_heavy,
        }
        state.save()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO Lab refresh complete."))
        self.stdout.write(f"Analyzed through draw ID: {latest_draw_id}")