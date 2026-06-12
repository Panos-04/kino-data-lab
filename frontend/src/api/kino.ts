import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000/api/kino";

export type WindowNumber = {
    number: number;
    count: number;
    percentage: number;
    first_half_count?: number;
    second_half_count?: number;
};

export type WindowAnalysis = {
    id: number;
    window_size: number;
    step_size: number;
    start_draw_id: number;
    end_draw_id: number;
    start_time: number;
    end_time: number;
    numbers: WindowNumber[];
};

export async function fetchWindows(
    windowSize: number,
    stepSize: number,
    limit: number = 50
): Promise<WindowAnalysis[]> {
    const response = await axios.get(`${API_BASE_URL}/windows/`, {
        params: {
            window_size: windowSize,
            step_size: stepSize,
            limit,
        },
    });

    return response.data.results;
}
export type RelatedNumber = {
    number: number;
    total_count: number;
    first_half_count: number;
    second_half_count: number;
    change: number;
};

export type NumberRelations = {
    window_id: number;
    window_size: number;
    step_size: number;
    start_draw_id: number;
    end_draw_id: number;
    selected_number: number;
    anchor_appearances: number;
    first_half_anchor_appearances: number;
    second_half_anchor_appearances: number;
    related_numbers: RelatedNumber[];
};

export async function fetchNumberRelations(
    windowId: number,
    number: number,
    top: number = 15
): Promise<NumberRelations> {
    const response = await axios.get(
        `${API_BASE_URL}/windows/${windowId}/relations/${number}/`,
        {
            params: {
                top,
            },
        }
    );

    return response.data;
} export type GeneralRelatedNumber = {
    number: number;
    total_count: number;
    first_half_count: number;
    second_half_count: number;
    change: number;
};

export type GeneralRelationAnchor = {
    anchor_number: number;
    anchor_type: "hot" | "cold" | "middle";
    anchor_heat: number;
    anchor_appearances: number;
    first_half_anchor_appearances: number;
    second_half_anchor_appearances: number;
    strongest_connections: GeneralRelatedNumber[];
    weakest_connections: GeneralRelatedNumber[];
};

export type GeneralRelations = {
    window_id: number;
    window_size: number;
    step_size: number;
    start_draw_id: number;
    end_draw_id: number;
    expected_heat: number;
    anchors: GeneralRelationAnchor[];
};

export async function fetchGeneralRelations(
    windowId: number,
    top: number = 20,
    bottom: number = 20
): Promise<GeneralRelations> {
    const response = await axios.get(
        `${API_BASE_URL}/windows/${windowId}/general-relations/`,
        {
            params: {
                top,
                bottom,
            },
        }
    );

    return response.data;
}

export type ComboDistributionRow = {
    hits: number;
    count: number;
    percentage: number;
    baseline_percentage: number;
    difference: number;
};

export type ComboBestResult = {
    window_id: number;
    draw_id: number;
    combo: number[];
    draw_numbers: number[];
    hit_count: number;
    hit_numbers: number[];
};

export type ComboTestResult = {
    strategy: "cold" | "hot" | "middle";
    window_size: number;
    step_size: number;
    pick_count: number;
    future_size: number;
    tested_draws: number;
    skipped_windows: number;
    distribution: ComboDistributionRow[];
    four_plus: {
        count: number;
        percentage: number;
        baseline_percentage: number;
        difference: number;
    };
    best_results: ComboBestResult[];
};

export async function fetchComboTest(params: {
    strategy: "cold" | "hot" | "middle";
    window: number;
    step: number;
    pick: number;
    future: number;
}): Promise<ComboTestResult> {
    const response = await axios.get(`${API_BASE_URL}/combo-test/`, {
        params,
    });

    return response.data;
}

export type PatternExample = {
    draw_id: number;
    draw_time: number;
    type: "row" | "column";
    group: number;
    hit_count: number;
    hit_numbers: number[];
    draw_numbers: number[];
};

export type PatternSummaryRow = {
    group: number;
    count: number;
    percentage: number;
};

export type PatternStreak = {
    type: "row" | "column";
    group: number;
    streak: number;
};

export type PatternTestResult = {
    total_draws: number;
    row_threshold: number;
    column_threshold: number;
    row_pattern_count: number;
    column_pattern_count: number;
    row_pattern_percentage: number;
    column_pattern_percentage: number;
    row_summary: PatternSummaryRow[];
    column_summary: PatternSummaryRow[];
    streaks: PatternStreak[];
    row_gap_summary: PatternGapSummary[];
    column_gap_summary: PatternGapSummary[];
    row_repeat_summary: PatternRepeatSummary[];
    column_repeat_summary: PatternRepeatSummary[];
    row_patterns: PatternExample[];
    column_patterns: PatternExample[];
};

export async function fetchPatternTest(params: {
    row_threshold: number;
    column_threshold: number;
    limit: number;
}): Promise<PatternTestResult> {
    const response = await axios.get(`${API_BASE_URL}/pattern-test/`, {
        params,
    });

    return response.data;
}

export type PatternGapExample = {
    from_draw_id: number;
    to_draw_id: number;
    gap: number;
};

export type PatternGapSummary = {
    type: "row" | "column";
    group: number;
    events: number;
    repeat_count: number;
    min_gap: number | null;
    max_gap: number | null;
    avg_gap: number | null;
    examples: PatternGapExample[];
};
export type PatternRepeatRate = {
    within_games: number;
    repeat_count: number;
    tested_events: number;
    repeat_rate: number;
};

export type PatternRepeatSummary = {
    type: "row" | "column";
    group: number;
    events: number;
    repeat_rates: PatternRepeatRate[];
};

export type ShapePatternExample = {
    shape: string;
    center_number: number;
    center_row: number;
    center_col: number;
    shape_numbers: number[];
    hit_numbers: number[];
    hit_count: number;
    shape_size: number;
    draw_id: number;
    draw_time: number;
    draw_numbers: number[];
};

export type ShapeSummary = {
    shape: string;
    events: number;
    draws_with_shape: number;
    draw_percentage: number;
    avg_events_per_draw: number;
};

export type ShapeCenterSummary = {
    shape: string;
    center_number: number;
    events: number;
};

export type ShapeHitCountRow = {
    hit_count: number;
    events: number;
};

export type ShapePatternTestResult = {
    shape: string;
    min_hits: number | null;
    total_draws: number;
    total_events: number;
    draws_with_any_shape: number;
    draws_with_any_shape_percentage: number;
    avg_events_per_draw: number;
    most_events_in_one_draw: number;
    shape_summary: ShapeSummary[];
    center_summary: ShapeCenterSummary[];
    hit_count_summary: Record<string, ShapeHitCountRow[]>;
    examples: ShapePatternExample[];
};

export async function fetchShapePatternTest(params: {
    shape: string;
    min_hits?: number;
    limit: number;
}): Promise<ShapePatternTestResult> {
    const response = await axios.get(`${API_BASE_URL}/shape-pattern-test/`, {
        params,
    });

    return response.data;
}

export type ShapeMovementVector = {
    delta_row: number;
    delta_col: number;
    count: number;
    percentage: number;
};

export type ShapeMovementGap = {
    gap: number;
    count: number;
    percentage: number;
};

export type ShapeMovementCenter = {
    from_center: number;
    to_center: number;
    count: number;
    percentage: number;
};

export type ShapeMovementExample = {
    id: number;
    from_draw_id: number;
    to_draw_id: number;
    from_center: number;
    to_center: number;
    delta_row: number;
    delta_col: number;
    gap: number;
    overlap_score: number;
    distance_score: number;
};

export type ShapeMovementsResult = {
    shape: string;
    mode: string;
    min_hits: number;
    future_window: number;
    total_movements: number;
    vector_summary: ShapeMovementVector[];
    gap_summary: ShapeMovementGap[];
    center_summary: ShapeMovementCenter[];
    examples: ShapeMovementExample[];
};

export async function fetchShapeMovements(params: {
    shape: string;
    mode: string;
    min_hits: number;
    future: number;
    limit: number;
}): Promise<ShapeMovementsResult> {
    const response = await axios.get(`${API_BASE_URL}/shape-movements/`, {
        params,
    });

    return response.data;
}

export type AINumberScore = {
    rank: number;
    number: number;
    row: number;
    column: number;
    probability: number;
    probability_percent: number;
    above_baseline: number;
    count_last_10: number;
    count_last_20: number;
    count_last_50: number;
};

export type AIFeatureImportance = {
    feature: string;
    coefficient: number;
    absolute_strength: number;
};

export type AIResultsResponse = {
    has_result: boolean;
    message?: string;
    id?: number;
    model_name?: string;
    train_draws?: number;
    test_draws?: number;
    baseline_top20_hits?: number;
    model_top20_hits?: number;
    lift?: number;
    accuracy?: number;
    precision?: number;
    recall?: number;
    created_at?: string;
    data?: {
        mode?: string;
        horizon?: number;
        decision_step?: number;
        pick?: number;
        target_hits?: number;
        baseline_target_probability?: number;

        latest_draw_id: number;
        split_draw_id: number;
        training_rows: number;
        testing_rows: number;

        top20_hits_by_test_draw?: number[];
        model_top20_hits_by_test_decision?: number[];
        random_top20_hits_by_test_decision?: number[];

        model_pick_hits_by_test_decision?: number[];
        random_pick_hits_by_test_decision?: number[];
        random_pick_average_hits?: number;

        latest_scores: AINumberScore[];
        top20_latest_scores: AINumberScore[];
        top_pick_latest_scores?: AINumberScore[];

        feature_importance: AIFeatureImportance[];
        model_path?: string | null;
        created_at: string;
        raw_profit_summary?: AIProfitSummary;
        spread_profit_summary?: AIProfitSummary;
        hybrid_profit_summary?: AIProfitSummary;
        miss_profit_summary?: AIProfitSummary;
        regime_profit_summary?: AIProfitSummary;
        random_profit_summary?: AIProfitSummary;

        latest_confidence?: {
            confidence_score: number;
            decision: string;
            reasons: string[];
            component_counts?: Record<string, number>;
            above_baseline_count?: number;
            strong_above_baseline_count?: number;
            unique_rows?: number;
            unique_columns?: number;
        };

        confidence_summary?: AIConfidenceSummary;

        audit_examples_by_quality?: {
            high_confidence_wins?: AIAuditExample[];
            high_confidence_losses?: AIAuditExample[];
            low_confidence_wins?: AIAuditExample[];
            low_confidence_losses?: AIAuditExample[];
        };

        latest_number_explanations?: AINumberExplanation[];

        best_mode?: string;
        latest_operation?: string;
        latest_zone?: string;
        latest_operation_streak_length?: number;
        latest_regime?: string;
        latest_selected_mode?: string;
        payout_table?: string;
        stake?: number;
        cost_per_combo_decision?: number;

        relaxed_hybrid_profit_summary?: AIProfitSummary;
        rescue_1_profit_summary?: AIProfitSummary;
        smart_rescue_1_profit_summary?: AIProfitSummary;
        rescue_2_profit_summary?: AIProfitSummary;
        rescue_3_profit_summary?: AIProfitSummary;

        rescue_comparison_summary?: Record<string, unknown>;
        smart_rescue_context_summary?: unknown[];
        smart_rescue_coarse_summary?: unknown[];
        safe_smart_rescue_1_profit_summary?: AIProfitSummary;


        latest_rescue_1_scores?: unknown[];
        latest_smart_rescue_1_scores?: unknown[];
        latest_smart_rescue_info?: Record<string, unknown>;


        swap_model_1_profit_summary?: AIProfitSummary;

        swap_model_training_summary?: {
            training_rows?: number;
            positive_swaps?: number;
            negative_swaps?: number;
            positive_rate?: number;
            accuracy?: number;
            precision?: number;
            recall?: number;
            applied_swaps?: number;
            skipped_swaps?: number;
            avg_predicted_probability?: number;
            best_probability_threshold?: number;
            feature_count?: number;
        };

        swap_model_feature_names?: string[];

        latest_swap_model_1_scores?: AILatestScore[];

        latest_swap_model_1_info?: {
            applied?: boolean;
            reason?: string;
            predicted_probability?: number;
            dropped_number?: number;
            added_number?: number;
            base_numbers?: number[];
            swapped_numbers?: number[];
            reserve_pool?: number[];
            candidate_count?: number;
            best_candidate?: {
                dropped_number?: number;
                added_number?: number;
                predicted_probability?: number;
                drop_rank?: number;
                add_rank?: number;
                drop_probability?: number;
                add_probability?: number;
            };
        };
    };
};
export type AIProfitSummary = {
    stake_per_round?: number;
    rounds_per_combo?: number;
    cost_per_combo_decision?: number;
    total_combo_decisions?: number;
    total_rounds_played?: number;
    total_cost?: number;
    total_return?: number;
    total_profit?: number;
    roi?: number;
    profitable_decisions?: number;
    losing_decisions?: number;
    break_even_decisions?: number;
    hit_distribution?: Record<string, number>;
    bonus_hit_distribution?: Record<string, number>;
    paying_rounds?: number;
    paying_round_rate?: number;
    dead_zone_rounds?: number;
    dead_zone_rate?: number;
    component_summary?: Record<
        string,
        {
            hits: number;
            misses: number;
            total: number;
            hit_rate: number;
        }
    >;
};
export type AILatestScore = {
    rank?: number;
    number: number;
    row?: number;
    column?: number;
    probability?: number;
    probability_percent?: number;
    above_baseline?: number;
    empirical_lift?: number;
    count_last_10?: number;
    count_last_20?: number;
    count_last_50?: number;
};
export type AIAuditRoundDetail = {
    future_draw_id: number;
    hit_count: number;
    hit_numbers: number[];
    bonus_number?: number | null;
    bonus_hit?: boolean;
    payout: number;
};

export type AINumberExplanation = {
    number: number;
    components: string[];
    reasons: string[];
    probability?: number;
    probability_percent?: number;
};

export type AIAuditExample = {
    draw_id: number;
    draw_index: number;
    selected_numbers: number[];
    selected_mode: string;
    regime: string;
    operation: string;
    zone: string;
    confidence_score: number;
    confidence_decision: string;
    confidence_reasons: string[];
    component_counts: Record<string, number>;
    cost: number;
    return: number;
    profit: number;
    roi: number;
    hit_distribution: Record<string, number>;
    bonus_hit_distribution?: Record<string, number>;
    round_details: AIAuditRoundDetail[];
    number_explanations?: AINumberExplanation[];
};

export type AIConfidenceSummary = {
    thresholds?: Record<
        string,
        {
            played_decisions: number;
            skipped_decisions: number;
            roi: number | null;
            profit: number | null;
            cost?: number;
            return?: number;
            paying_round_rate?: number;
            dead_zone_rate?: number;
            hit_distribution?: Record<string, number>;
        }
    >;
    buckets?: Record<
        string,
        {
            decisions: number;
            roi: number | null;
            profit: number | null;
            cost?: number;
            return?: number;
            paying_round_rate?: number;
            dead_zone_rate?: number;
        }
    >;
};
export async function fetchAIResults(): Promise<AIResultsResponse> {
    const response = await axios.get(`${API_BASE_URL}/ai-results/`);
    return response.data;
}