from tqdm.auto import tqdm


def is_streamlit():
    try:
        import streamlit as st

        return hasattr(st, "runtime") and st.runtime.exists()
    except ImportError:
        return False


class ProgressBar:
    def __init__(self, total=None, desc=None, unit=None, st_container=None):
        self.is_streamlit_env = is_streamlit()

        self.total = total
        self.desc = desc
        self.unit = unit
        self.st_container = st_container

        if self.is_streamlit_env:
            import streamlit as st

            self.st = st
            self.container = st_container or st.empty()
            self.progress = self.container.progress(0, text=desc)
            self.count = 0

            self.pbar_console = tqdm(total=total, desc=desc, unit=unit)
        else:
            self.pbar = tqdm(total=total, desc=desc, unit=unit)

    def update(self, n=1):
        if self.is_streamlit_env:
            self.count += n
            ratio = min(self.count / self.total, 1) if self.total else 0
            text = (
                f"{self.desc}: {self.count} / {self.total} {self.unit} ({ratio*100:.2f}%)"
                if self.total
                else f"{self.desc}: {self.count} {self.unit}"
            )
            self.progress.progress(ratio, text=text)
            self.pbar_console.update(n)
        else:
            self.pbar.update(n)

    def close(self):
        if self.is_streamlit_env:
            self.container.empty()
            self.pbar_console.close()
        else:
            self.pbar.close()
