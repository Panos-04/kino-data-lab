export type RelationSelection = {
    windowId: number;
    frameIndex: number;
    number: number;
    windowSize: number;
    stepSize: number;
  };
  
  const STORAGE_KEY = "kino_relation_selection";
  
  export function saveRelationSelection(selection: RelationSelection) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
  
    window.dispatchEvent(
      new CustomEvent("kino-relation-selection", {
        detail: selection,
      })
    );
  }
  
  export function getRelationSelection(): RelationSelection | null {
    const raw = localStorage.getItem(STORAGE_KEY);
  
    if (!raw) return null;
  
    try {
      return JSON.parse(raw) as RelationSelection;
    } catch {
      return null;
    }
  }
  
  export function getRelationStorageKey() {
    return STORAGE_KEY;
  }