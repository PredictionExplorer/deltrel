import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Fraunces } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  axes: ["opsz"],
});

const siteUrl = process.env.DELTREL_SITE_URL ?? (
  process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : `http://localhost:${process.env.PORT || '3000'}`
);
const metadataBase = new URL(siteUrl);
if (!['http:', 'https:'].includes(metadataBase.protocol)) {
  throw new Error('DELTREL_SITE_URL must be an absolute HTTP or HTTPS URL.');
}

export const metadata: Metadata = {
  metadataBase,
  title: "Deltrel — Connection runs deep",
  description:
    "Reach the shore. Join your networks. Deltrel is a thoughtful connection game for two, set where the river meets the sea. Play Classic or Double on four board sizes.",
  applicationName: "Deltrel",
  appleWebApp: { capable: true, title: "Deltrel", statusBarStyle: "default" },
  openGraph: {
    title: "Deltrel — Connection runs deep",
    description: "A game of connection. A world between you.",
    siteName: "Deltrel",
    type: "website",
    locale: "en_US",
  },
  twitter: { card: "summary_large_image", title: "Deltrel — Connection runs deep" },
};

export const viewport: Viewport = {
  themeColor: "#0b393c",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${fraunces.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
