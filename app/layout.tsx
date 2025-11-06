import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TDR Cable Fault Detection",
  description: "AI-powered Time Domain Reflectometry analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
