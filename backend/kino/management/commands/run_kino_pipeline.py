from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = "Run full KINO pipeline: sync, windows, patterns, shapes, movements, and AI training"

    def add_arguments(self, parser):
        parser.add_argument(
            "--skip-sync",
            action="store_true",
            help="Skip syncing latest KINO draws",
        )

        parser.add_argument(
            "--skip-heavy",
            action="store_true",
            help="Skip heavy pattern/shape/movement rebuilds",
        )

        parser.add_argument(
            "--skip-ai",
            action="store_true",
            help="Skip AI training at the end",
        )

        parser.add_argument(
            "--rebuild-shapes",
            action="store_true",
            help="Rebuild shape events from scratch",
        )

        parser.add_argument(
            "--rebuild-movements",
            action="store_true",
            help="Rebuild shape movements from scratch",
        )

        parser.add_argument(
            "--row-threshold",
            type=int,
            default=6,
        )

        parser.add_argument(
            "--column-threshold",
            type=int,
            default=5,
        )

        parser.add_argument(
            "--shape-min-hits",
            type=int,
            default=4,
        )

        parser.add_argument(
            "--movement-future",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--movement-mode",
            type=str,
            default="one-to-one",
        )

        parser.add_argument(
            "--ai-horizon",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--ai-decision-step",
            type=int,
            default=5,
        )

        parser.add_argument(
            "--ai-pick",
            type=int,
            default=12,
        )

        parser.add_argument(
            "--ai-target-hits",
            type=int,
            default=3,
        )

    def handle(self, *args, **options):
        started_at = timezone.now()

        def step(title):
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(f"▶ {title}"))

        def done(title):
            self.stdout.write(self.style.SUCCESS(f"✓ {title}"))

        def run_command(name, *cmd_args, **cmd_options):
            step(f"Running {name}...")

            call_command(
                name,
                *cmd_args,
                **cmd_options,
            )

            done(f"{name} finished")

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Starting KINO full pipeline"))
        self.stdout.write(f"Started at: {started_at}")

        # ------------------------------------------------------------
        # 1. Sync latest draws
        # ------------------------------------------------------------

        if not options["skip_sync"]:
            run_command("sync_kino_latest")
        else:
            self.stdout.write(self.style.WARNING("Skipping sync_kino_latest"))

        # ------------------------------------------------------------
        # 2. Build windows
        # ------------------------------------------------------------

        run_command("build_windows_incremental")

        # ------------------------------------------------------------
        # 3. Build board pattern events
        # ------------------------------------------------------------

        if not options["skip_heavy"]:
            run_command(
                "build_board_pattern_events",
                row_threshold=options["row_threshold"],
                column_threshold=options["column_threshold"],
            )
        else:
            self.stdout.write(self.style.WARNING("Skipping board pattern events"))

        # ------------------------------------------------------------
        # 4. Build shape events
        # ------------------------------------------------------------

        if not options["skip_heavy"]:
            run_command(
                "build_shape_events",
                shape="all",
                min_hits=options["shape_min_hits"],
                rebuild=options["rebuild_shapes"],
            )
        else:
            self.stdout.write(self.style.WARNING("Skipping shape events"))

        # ------------------------------------------------------------
        # 5. Build shape movements
        # ------------------------------------------------------------

        if not options["skip_heavy"]:
            run_command(
                "build_shape_movements",
                shape="all",
                min_hits=options["shape_min_hits"],
                future=options["movement_future"],
                mode=options["movement_mode"],
                rebuild=options["rebuild_movements"],
            )
        else:
            self.stdout.write(self.style.WARNING("Skipping shape movements"))

        # ------------------------------------------------------------
        # 6. Train AI model
        # ------------------------------------------------------------

        if not options["skip_ai"]:
            run_command(
                "train_number_ai_10game",
                horizon=options["ai_horizon"],
                decision_step=options["ai_decision_step"],
                pick=options["ai_pick"],
                target_hits=options["ai_target_hits"],
            )
        else:
            self.stdout.write(self.style.WARNING("Skipping AI training"))

        finished_at = timezone.now()
        duration = finished_at - started_at

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("KINO full pipeline completed"))
        self.stdout.write(f"Finished at: {finished_at}")
        self.stdout.write(f"Duration: {duration}")