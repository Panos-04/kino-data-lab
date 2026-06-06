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