from http.server import BaseHTTPRequestHandler, HTTPServer


PORT = 8000


HTML = r'''<!doctype html>
<html lang="es">
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Relay — Automatiza lo que mueve tu negocio</title>
	<meta name="description" content="Relay conecta tus herramientas y convierte procesos repetitivos en flujos que trabajan solos.">
	<style>
		@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

		:root {
			--ink: #17211f;
			--muted: #66716d;
			--paper: #f4f1ea;
			--surface: #fbfaf6;
			--line: #d8ded5;
			--coral: #ef6b51;
			--coral-dark: #c95642;
			--teal: #1f706b;
			--yellow: #f3ce61;
			--shadow: 0 20px 50px rgba(39, 55, 45, .12);
		}

		* { box-sizing: border-box; }
		html { scroll-behavior: smooth; }
		body { margin: 0; color: var(--ink); background: var(--paper); font-family: 'Space Grotesk', sans-serif; }
		a { color: inherit; text-decoration: none; }
		button { font: inherit; cursor: pointer; }
		.wrap { width: min(1160px, calc(100% - 40px)); margin: 0 auto; }
		.mono { font-family: 'DM Mono', monospace; letter-spacing: .03em; }

		nav { display: flex; align-items: center; justify-content: space-between; padding: 24px 0; }
		.brand { display: flex; align-items: center; gap: 10px; font-size: 1.2rem; font-weight: 700; }
		.brand-mark { display: grid; place-items: center; width: 29px; height: 29px; border-radius: 8px; color: var(--surface); background: var(--ink); font-weight: 700; }
		.nav-links { display: flex; gap: 32px; color: #53615b; font-size: .9rem; }
		.nav-links a:hover { color: var(--coral-dark); }
		.nav-actions { display: flex; align-items: center; gap: 22px; font-size: .9rem; }
		.text-button { border: 0; background: transparent; color: var(--ink); }
		.button { display: inline-flex; align-items: center; justify-content: center; gap: 9px; border: 0; border-radius: 5px; padding: 13px 18px; color: #fffdf7; background: var(--coral); font-weight: 600; box-shadow: 0 7px 0 #d55641; transition: transform .2s, box-shadow .2s, background .2s; }
		.button:hover { background: var(--coral-dark); transform: translateY(2px); box-shadow: 0 4px 0 #b84937; }
		.button.dark { background: var(--ink); box-shadow: 0 7px 0 #0d1413; }
		.button.dark:hover { background: #273633; box-shadow: 0 4px 0 #0d1413; }

		.hero { display: grid; grid-template-columns: .87fr 1.13fr; gap: 72px; align-items: center; padding: 78px 0 108px; }
		.eyebrow { display: flex; align-items: center; gap: 10px; color: var(--teal); font-size: .72rem; font-weight: 600; text-transform: uppercase; }
		.eyebrow::before { content: ''; width: 26px; height: 2px; background: var(--coral); }
		h1 { max-width: 590px; margin: 19px 0 20px; font-size: clamp(3.3rem, 6vw, 5.8rem); line-height: .96; letter-spacing: -.075em; font-weight: 600; }
		.hero-copy { max-width: 450px; color: var(--muted); font-size: 1.09rem; line-height: 1.6; }
		.hero-actions { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 31px; }
		.micro-note { margin-top: 21px; color: #77817c; font-size: .73rem; }
		.micro-note span { margin-right: 14px; }
		.micro-note span::before { content: '✓'; margin-right: 5px; color: var(--teal); font-weight: 700; }

		.workflow-frame { position: relative; min-height: 500px; padding: 26px; border: 1px solid #d5dcd4; border-radius: 13px; background: #e4e9e0; box-shadow: var(--shadow); overflow: hidden; }
		.workflow-frame::before { content: ''; position: absolute; inset: 0; opacity: .45; background-image: radial-gradient(#aebcb0 1px, transparent 1px); background-size: 19px 19px; }
		.workflow-header { position: relative; z-index: 1; display: flex; justify-content: space-between; color: #6c7971; font-size: .7rem; }
		.status { color: var(--teal); }
		.flow { position: relative; z-index: 1; display: grid; grid-template-columns: 1fr 42px 1fr 42px 1fr; align-items: center; margin-top: 87px; }
		.flow-line { height: 1px; background: var(--teal); position: relative; }
		.flow-line::after { content: '›'; position: absolute; right: -2px; top: -12px; color: var(--teal); font-size: 22px; }
		.node { min-height: 140px; padding: 17px; border: 1px solid #cbd4ca; border-radius: 9px; background: rgba(255, 255, 249, .93); box-shadow: 0 8px 19px rgba(56, 73, 59, .1); }
		.node-top { display: flex; align-items: center; justify-content: space-between; color: #79847d; font-size: .64rem; }
		.node-icon { display: grid; place-items: center; width: 35px; height: 35px; margin: 17px 0 10px; border-radius: 8px; color: white; font-size: 1.1rem; font-weight: 700; }
		.icon-mail { background: var(--coral); }
		.icon-ai { background: var(--teal); }
		.icon-sheet { background: #2c9878; }
		.node-title { font-size: .92rem; font-weight: 600; }
		.node-sub { margin-top: 4px; color: var(--muted); font-size: .68rem; }
		.flow-caption { position: absolute; bottom: 28px; left: 29px; color: #66746b; font-size: .68rem; }
		.pulse { display: inline-block; width: 7px; height: 7px; margin-right: 6px; border-radius: 50%; background: #43ae78; animation: pulse 1.9s infinite; }
		@keyframes pulse { 50% { box-shadow: 0 0 0 6px rgba(67, 174, 120, .14); } }

		.proof { display: flex; justify-content: space-between; align-items: center; gap: 32px; padding: 28px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); color: #78827b; font-size: .73rem; }
		.logos { display: flex; align-items: center; gap: clamp(20px, 4vw, 52px); color: var(--ink); font-size: 1rem; font-weight: 600; }
		.logo-dot { display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; background: var(--coral); }

		.section { padding: 125px 0; }
		.section-head { display: flex; justify-content: space-between; align-items: end; gap: 30px; margin-bottom: 46px; }
		h2 { max-width: 550px; margin: 0; font-size: clamp(2.2rem, 4vw, 3.6rem); line-height: 1; letter-spacing: -.06em; font-weight: 600; }
		.section-intro { max-width: 310px; margin: 0; color: var(--muted); line-height: 1.55; }
		.use-cases { display: grid; grid-template-columns: repeat(3, 1fr); gap: 13px; }
		.case { position: relative; min-height: 267px; padding: 25px; border: 1px solid var(--line); background: var(--surface); overflow: hidden; transition: transform .25s, border-color .25s; }
		.case:hover { border-color: var(--coral); transform: translateY(-5px); }
		.case:nth-child(2) { background: var(--yellow); border-color: #e8c65d; }
		.case:nth-child(3) { color: #f7fbf5; background: var(--teal); border-color: var(--teal); }
		.case-num { color: var(--coral-dark); font-size: .72rem; }
		.case:nth-child(3) .case-num { color: #a7dad0; }
		.case h3 { max-width: 190px; margin: 73px 0 10px; font-size: 1.35rem; line-height: 1.05; letter-spacing: -.04em; }
		.case p { max-width: 220px; margin: 0; color: var(--muted); font-size: .82rem; line-height: 1.5; }
		.case:nth-child(3) p { color: #bfdfd8; }
		.case-arrow { position: absolute; right: 23px; bottom: 22px; font-size: 1.5rem; }

		.cta { display: grid; grid-template-columns: 1fr .8fr; gap: 60px; align-items: center; margin-bottom: 110px; padding: 60px; color: #f8fbf4; background: var(--ink); }
		.cta h2 { max-width: 590px; }
		.cta p { max-width: 380px; color: #b8c4bd; line-height: 1.6; }
		.cta .button { margin-top: 19px; color: var(--ink); background: var(--yellow); box-shadow: 0 7px 0 #c9a746; }
		.cta .button:hover { background: #f7d875; box-shadow: 0 4px 0 #c9a746; }
		footer { display: flex; justify-content: space-between; padding: 23px 0 34px; border-top: 1px solid var(--line); color: #758079; font-size: .72rem; }
		footer div:last-child { display: flex; gap: 24px; }

		@media (max-width: 820px) {
			.nav-links, .nav-actions .text-button { display: none; }
			.hero { grid-template-columns: 1fr; gap: 48px; padding: 55px 0 76px; }
			.workflow-frame { min-height: 420px; }
			.flow { margin-top: 72px; transform: scale(.9); transform-origin: top center; width: 111%; margin-left: -5.5%; }
			.proof { align-items: flex-start; flex-direction: column; }
			.logos { flex-wrap: wrap; }
			.section { padding: 85px 0; }
			.section-head { display: block; }
			.section-intro { margin-top: 20px; }
			.use-cases { grid-template-columns: 1fr; }
			.case { min-height: 220px; }
			.case h3 { margin-top: 53px; }
			.cta { grid-template-columns: 1fr; gap: 10px; margin-bottom: 70px; padding: 35px 28px; }
		}
		@media (max-width: 500px) {
			.wrap { width: min(100% - 28px, 1160px); }
			h1 { font-size: 3.15rem; }
			.hero-actions .button { width: 100%; }
			.workflow-frame { min-height: 335px; padding: 18px; }
			.flow { margin-top: 53px; transform: scale(.67); width: 149%; margin-left: -24.5%; }
			.flow-caption { left: 18px; bottom: 19px; }
			footer { display: block; }
			footer div:last-child { margin-top: 13px; }
		}
	</style>
</head>
<body>
	<header class="wrap">
		<nav>
			<a class="brand" href="#top"><span class="brand-mark">r</span> relay</a>
			<div class="nav-links"><a href="#how">Cómo funciona</a><a href="#cases">Casos de uso</a><a href="#pricing">Precios</a></div>
			<div class="nav-actions"><button class="text-button" type="button">Iniciar sesión</button><a class="button" href="#start">Probar gratis <span>↗</span></a></div>
		</nav>
	</header>

	<main id="top">
		<section class="wrap hero">
			<div>
				<div class="eyebrow mono">Automatización sin fricción</div>
				<h1>Haz que el trabajo avance solo.</h1>
				<p class="hero-copy">Relay conecta tus herramientas, entiende tus procesos y convierte las tareas repetitivas en flujos que trabajan mientras tú piensas en lo siguiente.</p>
				<div class="hero-actions"><a class="button" href="#start">Crea tu primer flujo <span>↗</span></a><a class="button dark" href="#how">Ver cómo funciona <span>▶</span></a></div>
				<div class="micro-note"><span>Sin tarjeta</span><span>5 min para empezar</span></div>
			</div>
			<div class="workflow-frame" aria-label="Ejemplo de flujo automatizado">
				<div class="workflow-header mono"><span>FLUJO / LEAD-ROUTING-07</span><span class="status">● ACTIVO</span></div>
				<div class="flow">
					<article class="node"><div class="node-top mono"><span>TRIGGER</span><span>01</span></div><div class="node-icon icon-mail">✉</div><div class="node-title">Nuevo formulario</div><div class="node-sub">Typeform · Cada envío</div></article>
					<div class="flow-line"></div>
					<article class="node"><div class="node-top mono"><span>AGENT</span><span>02</span></div><div class="node-icon icon-ai">✦</div><div class="node-title">Calificar lead</div><div class="node-sub">Relay AI · Analiza intención</div></article>
					<div class="flow-line"></div>
					<article class="node"><div class="node-top mono"><span>ACTION</span><span>03</span></div><div class="node-icon icon-sheet">↗</div><div class="node-title">Actualizar CRM</div><div class="node-sub">HubSpot · Asignar equipo</div></article>
				</div>
				<div class="flow-caption mono"><span class="pulse"></span>Última ejecución hace 2 min · 0 errores</div>
			</div>
		</section>

		<section class="wrap proof mono"><span>YA AUTOMATIZAN CON RELAY</span><div class="logos"><span><i class="logo-dot"></i>northstar</span><span>▲ orbit</span><span>lumen.</span><span>morrow</span></div></section>

		<section class="wrap section" id="how">
			<div class="section-head"><h2>De la idea al flujo en minutos.</h2><p class="section-intro">La potencia de una plataforma técnica, con la claridad de una herramienta que todo el equipo puede usar.</p></div>
			<div class="use-cases" id="cases">
				<article class="case"><div class="case-num mono">01 / CONECTA</div><h3>Todo tu stack, en un mismo lugar.</h3><p>Más de 200 integraciones listas para conectar con tus herramientas favoritas.</p><span class="case-arrow">↘</span></article>
				<article class="case"><div class="case-num mono">02 / DISEÑA</div><h3>Flujos tan visuales como tu proceso.</h3><p>Arrastra, suelta y dale forma a la lógica que hace avanzar tu negocio.</p><span class="case-arrow">↘</span></article>
				<article class="case"><div class="case-num mono">03 / ESCALA</div><h3>Automatiza con confianza.</h3><p>Observa cada ejecución, corrige lo que importa y crece sin añadir complejidad.</p><span class="case-arrow">↘</span></article>
			</div>
		</section>

		<section class="wrap cta" id="start">
			<div><div class="eyebrow mono">Tu próximo paso</div><h2>Menos pestañas abiertas. Más cosas terminadas.</h2></div>
			<div><p>Empieza gratis y descubre cuánto tiempo puede devolverte un buen flujo de trabajo.</p><a class="button" href="mailto:hola@relay.example">Empezar con Relay <span>↗</span></a></div>
		</section>
	</main>

	<footer class="wrap"><div>© 2026 Relay Systems</div><div><a href="#top">Privacidad</a><a href="#top">Contacto</a><a href="#top">Estado ↗</a></div></footer>
</body>
</html>'''


class LandingHandler(BaseHTTPRequestHandler):
		def do_GET(self):
				if self.path not in ('/', '/index.html'):
						self.send_error(404)
						return
				content = HTML.encode('utf-8')
				self.send_response(200)
				self.send_header('Content-Type', 'text/html; charset=utf-8')
				self.send_header('Content-Length', str(len(content)))
				self.end_headers()
				self.wfile.write(content)

		def log_message(self, format, *args):
				return


if __name__ == '__main__':
		server = HTTPServer(('127.0.0.1', PORT), LandingHandler)
		print(f'Relay landing disponible en http://127.0.0.1:{PORT}')
		try:
				server.serve_forever()
		except KeyboardInterrupt:
				print('\nServidor detenido.')
				server.server_close()
