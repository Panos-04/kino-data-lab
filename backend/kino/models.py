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
    
class KinoShapeEvent(models.Model):
    draw = models.ForeignKey(
        KinoDraw,
        on_delete=models.CASCADE,
        related_name="shape_events"
    )

    shape = models.CharField(max_length=40)

    center_number = models.IntegerField()
    center_row = models.IntegerField()
    center_col = models.IntegerField()

    shape_numbers = models.JSONField()
    hit_numbers = models.JSONField()

    hit_count = models.IntegerField()
    shape_size = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["shape"]),
            models.Index(fields=["center_number"]),
            models.Index(fields=["draw"]),
            models.Index(fields=["shape", "center_number"]),
        ]
        unique_together = (
            "draw",
            "shape",
            "center_number",
            "hit_count",
        )

    def __str__(self):
        return (
            f"{self.shape} center {self.center_number} "
            f"draw {self.draw.draw_id} {self.hit_count}/{self.shape_size}"
        )


class KinoShapeMovement(models.Model):
    from_event = models.ForeignKey(
        KinoShapeEvent,
        on_delete=models.CASCADE,
        related_name="movements_from"
    )

    to_event = models.ForeignKey(
        KinoShapeEvent,
        on_delete=models.CASCADE,
        related_name="movements_to"
    )

    shape = models.CharField(max_length=40)

    from_draw_id = models.BigIntegerField()
    to_draw_id = models.BigIntegerField()

    from_center = models.IntegerField()
    to_center = models.IntegerField()

    delta_row = models.IntegerField()
    delta_col = models.IntegerField()
    gap = models.IntegerField()

    overlap_score = models.IntegerField(default=0)
    distance_score = models.IntegerField(default=0)

    mode = models.CharField(max_length=30, default="one-to-one")
    future_window = models.IntegerField(default=10)
    min_hits = models.IntegerField(default=4)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["shape"]),
            models.Index(fields=["delta_row", "delta_col"]),
            models.Index(fields=["gap"]),
            models.Index(fields=["from_center", "to_center"]),
            models.Index(fields=["mode"]),
        ]
        unique_together = (
            "from_event",
            "to_event",
            "mode",
            "future_window",
            "min_hits",
        )

    def __str__(self):
        return (
            f"{self.shape}: {self.from_center} → {self.to_center} "
            f"Δr {self.delta_row:+}, Δc {self.delta_col:+}, gap {self.gap}"
        )
    
class KinoAnalysisState(models.Model):
    key = models.CharField(max_length=100, unique=True)
    value = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.key
    
class KinoBoardPatternEvent(models.Model):
    PATTERN_TYPES = (
        ("row", "Row"),
        ("column", "Column"),
    )

    draw = models.ForeignKey(
        KinoDraw,
        on_delete=models.CASCADE,
        related_name="board_pattern_events"
    )

    pattern_type = models.CharField(max_length=20, choices=PATTERN_TYPES)

    # row number 1-8 or column number 1-10
    group_number = models.IntegerField()

    group_numbers = models.JSONField()
    hit_numbers = models.JSONField()

    hit_count = models.IntegerField()
    threshold = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["pattern_type"]),
            models.Index(fields=["group_number"]),
            models.Index(fields=["hit_count"]),
            models.Index(fields=["threshold"]),
            models.Index(fields=["draw"]),
            models.Index(fields=["pattern_type", "group_number"]),
        ]

        unique_together = (
            "draw",
            "pattern_type",
            "group_number",
            "threshold",
        )

        ordering = ["draw__draw_time", "pattern_type", "group_number"]

    def __str__(self):
        return (
            f"{self.pattern_type} {self.group_number} "
            f"draw {self.draw.draw_id} "
            f"{self.hit_count}/{len(self.group_numbers)}"
        )
    
class KinoAIResult(models.Model):
    model_name = models.CharField(max_length=100, default="number_ai_v1")

    train_draws = models.IntegerField(default=0)
    test_draws = models.IntegerField(default=0)

    baseline_top20_hits = models.FloatField(default=5.0)
    model_top20_hits = models.FloatField(default=0)
    lift = models.FloatField(default=0)

    accuracy = models.FloatField(default=0)
    precision = models.FloatField(default=0)
    recall = models.FloatField(default=0)

    data = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"{self.model_name} | "
            f"top20 {self.model_top20_hits:.3f} | "
            f"lift {self.lift:.3f}"
        )