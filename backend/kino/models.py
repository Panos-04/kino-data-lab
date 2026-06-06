from django.db import models


class KinoDraw(models.Model):
    draw_id = models.BigIntegerField(unique=True)
    draw_time = models.BigIntegerField()
    numbers = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-draw_time"]

    def __str__(self):
        return f"KINO Draw {self.draw_id}"
    
class KinoWindowAnalysis(models.Model):
    window_size = models.IntegerField()
    step_size = models.IntegerField()

    start_draw = models.ForeignKey(
        KinoDraw,
        on_delete=models.CASCADE,
        related_name="window_starts"
    )
    end_draw = models.ForeignKey(
        KinoDraw,
        on_delete=models.CASCADE,
        related_name="window_ends"
    )

    start_time = models.BigIntegerField()
    end_time = models.BigIntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (
            "window_size",
            "step_size",
            "start_draw",
            "end_draw",
        )
        ordering = ["start_time"]

    def __str__(self):
        return (
            f"Window {self.window_size}/{self.step_size}: "
            f"{self.start_draw.draw_id} → {self.end_draw.draw_id}"
        )


class KinoWindowNumber(models.Model):
    analysis = models.ForeignKey(
        KinoWindowAnalysis,
        on_delete=models.CASCADE,
        related_name="numbers"
    )

    number = models.IntegerField()
    count = models.IntegerField()
    percentage = models.FloatField()

    class Meta:
        unique_together = ("analysis", "number")
        ordering = ["number"]

    def __str__(self):
        return f"{self.analysis_id} - Number {self.number}: {self.count}"
    
class KinoWindowAnchor(models.Model):
    analysis = models.ForeignKey(
        KinoWindowAnalysis,
        on_delete=models.CASCADE,
        related_name="anchors"
    )

    number = models.IntegerField()
    anchor_type = models.CharField(max_length=20)  # hot, cold, middle
    heat = models.IntegerField()

    anchor_appearances = models.IntegerField(default=0)
    first_half_anchor_appearances = models.IntegerField(default=0)
    second_half_anchor_appearances = models.IntegerField(default=0)

    class Meta:
        unique_together = ("analysis", "number")
        ordering = ["anchor_type", "-heat", "number"]

    def __str__(self):
        return f"{self.analysis_id} - {self.number} ({self.anchor_type})"


class KinoWindowRelation(models.Model):
    anchor = models.ForeignKey(
        KinoWindowAnchor,
        on_delete=models.CASCADE,
        related_name="relations"
    )

    related_number = models.IntegerField()

    total_count = models.IntegerField(default=0)
    first_half_count = models.IntegerField(default=0)
    second_half_count = models.IntegerField(default=0)
    change = models.IntegerField(default=0)

    class Meta:
        unique_together = ("anchor", "related_number")
        ordering = ["-total_count", "related_number"]

    def __str__(self):
        return f"{self.anchor.number} → {self.related_number}: {self.total_count}"