import { useEffect } from "react";

function usePageTitle(title: string) {
  useEffect(() => {
    document.title = `${title} | KINO Data Lab`;
  }, [title]);
}

export default usePageTitle;