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