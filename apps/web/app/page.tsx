import { ChatInterface } from "@/components/ChatInterface";

const W3C_MEMBER_LOGO = "https://www.w3.org/assets/logos/w3c-2025/sub-brands/svg/member.svg";

export default function Page() {
  return (
    <>
      <header className="site-header">
        <a className="brand" href="https://www.w3.org/" target="_blank" rel="noreferrer">
          <img src={W3C_MEMBER_LOGO} alt="W3C Member" />
        </a>
        <nav aria-label="Primary">
          <a href="https://www.w3.org/policies/process/" target="_blank" rel="noreferrer">
            Process
          </a>
          <a href="https://www.w3.org/guide/" target="_blank" rel="noreferrer">
            Guidebook
          </a>
          <a href="https://github.com/w3c/process" target="_blank" rel="noreferrer">
            Repository
          </a>
        </nav>
      </header>
      <ChatInterface />
      <footer className="site-footer">
        <p>For W3C Process guidance only. High-risk process questions should be confirmed with the relevant Staff Contact or W3C Team.</p>
      </footer>
    </>
  );
}
