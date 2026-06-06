from collections import Counter
from math import sqrt

from django.core.management.base import BaseCommand
from kino.models import KinoDraw


class Command(BaseCommand):
    help = "Analyze frequency of KINO numbers from imported draws"

    def handle(self, *args, **options):
        draws = KinoDraw.objects.all()

        if not draws.exists():
            self.stdout.write(self.style.WARNING("No draws found. Import data first."))
            return

        counter = Counter()
        total_draws = draws.count()

        for draw in draws:
            counter.update(draw.numbers)

        probability = 20 / 80
        expected = total_draws * probability
        std_dev = sqrt(total_draws * probability * (1 - probability))

        results = []

        for number in range(1, 81):
            observed = counter.get(number, 0)
            percentage = (observed / total_draws) * 100
            difference = observed - expected
            z_score = difference / std_dev if std_dev else 0

            results.append({
                "number": number,
                "observed": observed,
                "percentage": percentage,
                "difference": difference,
                "z_score": z_score,
            })

        results_by_frequency = sorted(
            results,
            key=lambda item: item["observed"],
            reverse=True
        )

        results_by_z = sorted(
            results,
            key=lambda item: abs(item["z_score"]),
            reverse=True
        )

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Analyzed {total_draws} draws"))
        self.stdout.write(f"Expected appearances per number: {expected:.2f}")
        self.stdout.write(f"Standard deviation: {std_dev:.2f}")
        self.stdout.write("")

        self.stdout.write("Top 10 most frequent numbers:")
        for item in results_by_frequency[:10]:
            self.stdout.write(
                f"{item['number']}: "
                f"{item['observed']} times "
                f"({item['percentage']:.2f}%), "
                f"diff {item['difference']:+.2f}, "
                f"z {item['z_score']:+.2f}"
            )

        self.stdout.write("")
        self.stdout.write("Bottom 10 least frequent numbers:")
        for item in results_by_frequency[-10:]:
            self.stdout.write(
                f"{item['number']}: "
                f"{item['observed']} times "
                f"({item['percentage']:.2f}%), "
                f"diff {item['difference']:+.2f}, "
                f"z {item['z_score']:+.2f}"
            )

        self.stdout.write("")
        self.stdout.write("Most statistically unusual numbers:")
        for item in results_by_z[:10]:
            self.stdout.write(
                f"{item['number']}: "
                f"{item['observed']} times, "
                f"diff {item['difference']:+.2f}, "
                f"z {item['z_score']:+.2f}"
            )