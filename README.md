# [Studio Name] — landing page

A bilingual (English/Spanish) landing page for a South Florida branding and social media studio, built from the Landing Page PRD. Next.js App Router + TypeScript + Tailwind CSS v4 + shadcn structure.

## Setup (on your computer)

You need Node.js 20 or newer.

```bash
npx create-next-app@latest studio-site --typescript --tailwind --eslint --app --import-alias "@/*"
cd studio-site
npx shadcn@latest init          # creates components.json, lib/utils.ts, installs tw-animate-css
npm install lucide-react
```

Copy `app/`, `components/` and `lib/` from this folder into the new project (replace the files it created), then:

```bash
npm run dev
```

Open http://localhost:3000.

## Before you publish

| What | Where |
| --- | --- |
| Studio name, WhatsApp number, email | `lib/site.ts` |
| Form destination (Formspree, Web3Forms, your API) | `NEXT_PUBLIC_FORM_ENDPOINT` in `.env.local` — without it, the form shows success but sends nothing |
| Real client stories, quotes and logos (with permission) | `lib/i18n.ts` → `proof`, and the logo row in `components/sections/hero.tsx` |
| Makeover video URL | `VIDEO_SRC` in `components/sections/video.tsx` |
| Social profile links | `components/sections/footer.tsx` |
| Privacy policy and terms pages | create `app/privacy/page.tsx` and `app/terms/page.tsx` |
| Spanish review by a native speaker | `lib/i18n.ts` → `es` |
| Analytics | add the GA4 or Plausible script in `app/layout.tsx`; events already fire through `track()` |

## Structure

```
app/
  layout.tsx            fonts, SEO metadata, LocalBusiness JSON-LD, SiteProvider
  page.tsx              section order (PRD sections 1–12)
  globals.css           Tailwind v4 + brand tokens + dark mode
components/
  site-provider.tsx     language (EN/ES), check-up form state, track() analytics helper
  ui/                   shadcn-style primitives
    display-cards.tsx, marketing-badges.tsx, before-after-slider.tsx, reveal.tsx
  sections/             one file per page section
    navbar, hero, problem, solution, services (+ offer-cards), cost-calculator,
    social-proof, pricing (+ FAQ), platforms, video, final-cta, footer, checkup-form
lib/
  i18n.ts               every word on the page, in English and Spanish
  calculator.ts         "cost of looking homemade" formula
  site.ts               studio settings and WhatsApp link helper
  styles.ts             shared button/section classes
  utils.ts              cn()
```

## Why `components/ui`

The shadcn CLI installs components into the folder named by the `ui` alias in `components.json` (`@/components/ui`), and components import each other from that path. Keeping primitives there and page sections in `components/sections` keeps imports working when you run `npx shadcn add`.

## Notes

- **Language:** follows the visitor's browser (Spanish browsers see Spanish), remembers their choice, and sets `<html lang>`.
- **Accessibility:** keyboard-operable badges, slider, accordion, pricing toggle and mobile menu; form errors announced; reduced motion respected; white text on gradients sits on a dark overlay for contrast.
- **No stock photos:** visuals are built in CSS, so the page loads fast on phones.
- **Placeholders are labeled on the page.** The social proof section says it's a placeholder until real client content replaces it.
- This build couldn't be run in the environment it was written in (the npm registry was blocked). Files were syntax-checked only, so run `npm run dev` and `npm run build` once and fix anything they flag.
