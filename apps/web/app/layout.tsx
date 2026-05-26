import "@w3c-process/ui/tokens.css";
import "./styles.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "W3C Process Assistant",
  description: "Internal assistant for W3C Process and standards workflow questions",
  icons: {
    icon: "https://www.w3.org/assets/logos/w3c-2025/sub-brands/svg/member.svg"
  }
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
