const APP_LINKS = [
    {
        label: "Trend Frames",
        href: "/trends",
    },
    {
        label: "Single Relations",
        href: "/single-number-relations",
    },
    {
        label: "General Relations",
        href: "/general-relations",
    },
    {
        label: "Combo Testing",
        href: "/combo-testing",
    },
    {
        label: "Pattern Testing",
        href: "/pattern-testing",
    },
    {
        label: "Shape Patterns",
        href: "/shape-pattern-testing",
    },
    {
        label: "Shape Movements",
        href: "/shape-movements",
    },
    {
        label: "AI Results",
        href: "/ai-results",
    },
];

function AppTopMenu() {
    return (
        <nav className="app-top-menu">
            <div className="app-top-menu-inner">
                <a
                    className="app-logo-link"
                    href="/trends"
                    target="_blank"
                    rel="noreferrer"
                >
                    KINO Data Lab
                </a>

                <div className="app-menu-links">
                    {APP_LINKS.map((link) => (
                        <a
                            key={link.href}
                            href={link.href}
                            target="_blank"
                            rel="noreferrer"
                            className="app-menu-link"
                        >
                            {link.label}
                        </a>
                    ))}
                </div>
            </div>
        </nav>
    );
}

export default AppTopMenu;