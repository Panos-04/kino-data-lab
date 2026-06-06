export type FrameSelection = {
    windowId: number;
    frameIndex: number;
    windowSize: number;
    stepSize: number;
    startDrawId: number;
    endDrawId: number;
};

const STORAGE_KEY = "kino_frame_selection";

export function saveFrameSelection(selection: FrameSelection) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));

    window.dispatchEvent(
        new CustomEvent("kino-frame-selection", {
            detail: selection,
        })
    );
}

export function getFrameSelection(): FrameSelection | null {
    const raw = localStorage.getItem(STORAGE_KEY);

    if (!raw) return null;

    try {
        return JSON.parse(raw) as FrameSelection;
    } catch {
        return null;
    }
}

export function getFrameStorageKey() {
    return STORAGE_KEY;
}