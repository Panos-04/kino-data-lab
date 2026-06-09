from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone

from kino.models import KinoDraw


class Command(BaseCommand):
    help = "Run full KINO analysis pipeline sequentially"

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-sync",
            action="store_true",
            help="Skip importing latest draws",
        )

        parser.add_argument(
            "--skip-heavy",
            action="store_true",
            help="Skip shape events and shape movements",
        )

        parser.add_argument(
            "--skip-reports",
            action="store_true",
            help="Skip terminal report commands like combo/pattern tests",
        )

        parser.add_argument(
            "--rebuild-shapes",
            action="store_true",
            help="Rebuild shape events from scratch",
        )

        parser.add_argument(
            "--rebuild-movements",
            action="store_true",
            help="Rebuild shape movement rows from scratch",
        )

    def handle(self, *args, **options):
        skip_sync = options["skip_sync"]
        skip_heavy = options["skip_heavy"]
        skip_reports = options["skip_reports"]
        rebuild_shapes = options["rebuild_shapes"]
        rebuild_movements = options["rebuild_movements"]

        started_at = timezone.now()

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("==================================="))
        self.stdout.write(self.style.WARNING("Running full KINO analysis pipeline"))
        self.stdout.write(self.style.WARNING("==================================="))

        latest_before = KinoDraw.objects.order_by("-draw_id").first()

        if latest_before:
            self.stdout.write(f"Latest draw before: {latest_before.draw_id}")
        else:
            self.stdout.write("Latest draw before: none")

        # 1. Sync latest draws
        if not skip_sync:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Step 1: Sync latest draws"))
            call_command("sync_kino_latest")
        else:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Step 1 skipped: Sync latest draws"))

        # 2. Build windows
        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Step 2: Build 20/10 windows"))
        call_command("build_windows_incremental", window=20, step=10)

        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Step 3: Build 10/5 windows"))
        call_command("build_windows_incremental", window=10, step=5)
        self.stdout.write("")
        self.stdout.write(self.style.WARNING("Step 4: Build board pattern events"))
        call_command(
            "build_board_pattern_events",
            row_threshold=6,
            column_threshold=5,
        )
        # 3. Heavy shape analysis
        if not skip_heavy:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Step 4: Build shape events"))

            if rebuild_shapes:
                call_command("build_shape_events", shape="all", rebuild=True)
            else:
                call_command("build_shape_events", shape="all")

            movement_shapes = [
                "cross",
                "box_2x2",
                "vertical_4",
                "horizontal_4",
                "diagonal_down_4",
                "diagonal_up_4",
                "l_shape",
            ]

            step_number = 5

            for shape in movement_shapes:
                self.stdout.write("")
                self.stdout.write(
                    self.style.WARNING(
                        f"Step {step_number}: Build shape movements for {shape}"
                    )
                )

                if rebuild_movements:
                    call_command(
                        "build_shape_movements",
                        shape=shape,
                        min_hits=4,
                        future=10,
                        mode="one-to-one",
                        rebuild=True,
                    )
                else:
                    call_command(
                        "build_shape_movements",
                        shape=shape,
                        min_hits=4,
                        future=10,
                        mode="one-to-one",
                    )

                step_number += 1
        else:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Heavy shape analysis skipped"))

        # 4. Reports / terminal tests
        if not skip_reports:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Running report commands"))

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report: Row/column board patterns"))
            call_command(
                "test_board_patterns",
                row_threshold=6,
                column_threshold=5,
                limit_results=10,
            )

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report: Cold combo hits"))
            call_command(
                "test_combo_hits",
                strategy="cold",
                window=20,
                step=10,
                pick=5,
                future=1,
            )

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report: Hot combo hits"))
            call_command(
                "test_combo_hits",
                strategy="hot",
                window=20,
                step=10,
                pick=5,
                future=1,
            )

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report: Middle combo hits"))
            call_command(
                "test_combo_hits",
                strategy="middle",
                window=20,
                step=10,
                pick=5,
                future=1,
            )

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report: Shape patterns"))
            call_command(
                "test_shape_patterns",
                shape="all",
                limit=10,
            )

            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report: Cross movement check"))
            call_command(
                "test_shape_movements",
                shape="cross",
                min_hits=4,
                future=10,
                mode="one-to-one",
                limit=10,
            )
        else:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("Report commands skipped"))

        latest_after = KinoDraw.objects.order_by("-draw_id").first()
        finished_at = timezone.now()
        duration = finished_at - started_at

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("==================================="))
        self.stdout.write(self.style.SUCCESS("KINO pipeline finished"))
        self.stdout.write(self.style.SUCCESS("==================================="))

        if latest_after:
            self.stdout.write(f"Latest draw after: {latest_after.draw_id}")

        self.stdout.write(f"Duration: {duration}")