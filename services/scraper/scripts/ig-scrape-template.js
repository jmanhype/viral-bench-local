// ego-browser Instagram post scraper
// Shortcode and URL are set as globals before the js() call

const shortcode = '__SHORTCODE__'
const navUrl = '__NAV_URL__'
const taskName = 'ig-scrape-' + shortcode

const task = await useOrCreateTaskSpace(taskName)

try {
  await openOrReuseTab(navUrl, { wait: true, timeout: 30 })
} catch(e) {
  cliLog(JSON.stringify({ success: false, error: 'Navigation failed: ' + e.message }))
  try { await completeTaskSpace(task.id, { keep: false }) } catch(x) {}
  process.exit(1)
}

await wait(5)

// Handle any dialogs
const pi = await pageInfo()
if (pi.dialog) {
  await cdp('Page.handleJavaScriptDialog', { accept: true })
  await wait(2)
}

// Check for error page
const bodyCheck = await js('document.body.innerText.substring(0, 500)')
if (bodyCheck.includes("Sorry, this page isn't available")) {
  cliLog(JSON.stringify({ success: false, error: 'Post not found or removed' }))
  try { await completeTaskSpace(task.id, { keep: false }) } catch(x) {}
  process.exit(1)
}

// Set shortcode in page context so js() can access it
await js('window.__IG_SHORTCODE = "' + shortcode + '"')

// Extract all metadata from DOM
const metadata = await js(String.raw`(() => {
  const sc = window.__IG_SHORTCODE || ''
  const result = {
    shortcode: sc,
    url: window.location.href,
    username: null,
    caption: null,
    like_count: null,
    comment_count: null,
    view_count: null,
    timestamp: null,
    media_type: null,
    images: [],
    videos: [],
    is_video: false
  }

  // Username - look for the post author link specifically
  // IG puts the username in a link right before the timestamp or in header
  const allLinks = document.querySelectorAll('a[href^="/"]')
  for (const l of allLinks) {
    const href = l.getAttribute('href') || ''
    // Match /username/ but not /p/, /reel/, /explore/, /direct/, etc.
    if (/^\/[a-zA-Z0-9_.]+\/$/.test(href) && 
        !href.includes('/p/') && !href.includes('/reel/') && 
        !href.includes('/explore/') && !href.includes('/direct/') &&
        !href.includes('/stories/') && !href.includes('/accounts/') &&
        href !== '/reels/' && href !== '/messages/' && href !== '/' &&
        !['/reels/', '/messages/', '/explore/', '/notifications/', '/accounts/'].includes(href)) {
      // Verify this link is near the post content (in header or article area)
      const parent = l.closest('header') || l.closest('article') || l.parentElement
      if (parent) {
        result.username = href.slice(1, -1)
        break
      }
    }
  }

  // Caption - find span that contains actual post text, not UI chrome
  // Strategy: look for spans inside article/main content area, skip footer/nav
  const article = document.querySelector('article') || document.querySelector('main')
  const searchRoot = article || document.body
  const spans = searchRoot.querySelectorAll('span')
  let bestCaption = ''
  for (const s of spans) {
    const text = (s.innerText || '').trim()
    // Skip short strings, UI chrome, and footer content
    if (text.length < 10 || text.length > 3000) continue
    if (text.includes('Cookie') || text.includes('Sign up') || text.includes('Log in')) continue
    if (text.includes('Afrikaans') || text.includes('English (UK)')) continue  // language selector
    if (text.includes('Meta') && text.includes('About') && text.includes('Blog')) continue  // footer
    // Prefer text that looks like a caption (contains sentences, hashtags, mentions)
    const hasContent = /[@#]/.test(text) || /\w+\s+\w+/.test(text)
    if (hasContent && text.length > bestCaption.length) {
      bestCaption = text
    }
  }
  if (bestCaption.length > 10) {
    // Strip IG UI chrome prefix: "instagram\n \n1h\n" or similar
    result.caption = bestCaption
      .replace(/^instagram\s*\n[\s\S]*?\n\d+[hdms]\s*\n/, '')  // "instagram ... 1h\n"
      .replace(/^instagram\s*\n/, '')  // just "instagram\n"
      .trim()
  }

  // Like count - search within article/main area first, then body
  const searchText = (document.querySelector('article') || document.body).innerText
  const likeMatch = searchText.match(/(\d[\d,.]*)\s*(likes?|others?)/i)
  if (likeMatch) result.like_count = parseInt(likeMatch[1].replace(/,/g, ''))

  // Comment count
  const commentMatch = searchText.match(/(\d[\d,.]*)\s*comments?/i)
  if (commentMatch) result.comment_count = parseInt(commentMatch[1].replace(/,/g, ''))

  // View count (reels)
  const viewMatch = searchText.match(/(\d[\d,.]*)\s*views?/i)
  if (viewMatch) result.view_count = parseInt(viewMatch[1].replace(/,/g, ''))

  // Timestamp
  const timeEl = document.querySelector('time')
  if (timeEl) result.timestamp = timeEl.getAttribute('datetime') || timeEl.innerText

  // Images from CDN
  const imgs = document.querySelectorAll('img')
  const seen = new Set()
  for (const img of imgs) {
    const src = img.src || ''
    if (src.includes('cdninstagram.com') && !src.includes('emoji') && !seen.has(src)) {
      seen.add(src)
      result.images.push(src.split('?')[0])
    }
  }

  // Video detection
  const vids = document.querySelectorAll('video')
  result.is_video = vids.length > 0
  result.media_type = result.is_video ? 'video' : (result.images.length > 1 ? 'carousel' : 'image')

  return result
})()`)

metadata.success = true
try { await completeTaskSpace(task.id, { keep: false }) } catch(x) {}
cliLog(JSON.stringify(metadata))
